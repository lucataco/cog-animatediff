# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
sys.path.extend(['/AnimateDiff'])
import tempfile
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from safetensors import safe_open
import torch
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora

from diffusers.utils.import_utils import is_xformers_available

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pretrained_model_path = "/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5"
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    def predict(
        self,
        motion_module: str = Input(
            description="Select a Motion Model",
            default="mm_sd_v14",
            choices=[
                "mm_sd_v14",
                "mm_sd_v15",
                "mm_sd_v15_v2"
            ],
        ),
        path: str = Input(
            default="toonyou_beta3.safetensors",
            choices=[
                "toonyou_beta3.safetensors",
                "lyriel_v16.safetensors",
                "rcnzCartoon3d_v10.safetensors",
                "majicmixRealistic_v5Preview.safetensors",
                "realisticVisionV40_v20Novae.safetensors"
            ],
            description="Select a Module",
        ),
        prompt: str = Input(description="Input prompt", default="masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, white hair, black eyes"),
        n_prompt: str = Input(description="Negative prompt", default=""),
        steps: int = Input(description="Number of inference steps", ge=1, le=100, default=25),
        guidance_scale: float = Input(description="guidance scale", ge=1, le=10, default=7.5),
        seed: int = Input(description="Seed (0 = random, maximum: 2147483647)", default=None),
        mp4: bool = Input(description="Produce .mp4 if true, or .gif if false", default=True)
    ) -> Path:
        """Run a single prediction on the model"""
        lora_alpha=0.8
        base=""
        # Create paths and load motion model
        newPath = "models/DreamBooth_LoRA/"+path
        motion_path = "/AnimateDiff/models/Motion_Module/"+motion_module+".ckpt"
        # Support new v2 motion module
        if motion_module.endswith("v2"):
            inference_config_file = "/AnimateDiff/configs/inference/inference-v2.yaml"
        else:
            inference_config_file = "/AnimateDiff/configs/inference/inference-v1.yaml"
        # Load configuration
        inference_config = OmegaConf.load(inference_config_file)
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            "/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5",
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
        )
        self.pipeline = AnimationPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")
        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

        motion_module_state_dict = torch.load(motion_path, map_location="cpu")
        missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0

        if path != "":
            fullPath = "/AnimateDiff/"+newPath
            if path.endswith(".ckpt"):
                state_dict = torch.load(fullPath)
                self.unet.load_state_dict(state_dict)

            elif path.endswith(".safetensors"):
                state_dict = {}
                base_state_dict = {}
                with safe_open(fullPath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

                is_lora = all("lora" in k for k in state_dict.keys())
                if not is_lora:
                    base_state_dict = state_dict
                else:
                    base_state_dict = {}
                    with safe_open(base, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            base_state_dict[key] = f.get_tensor(key)
                # vae
                converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, self.vae.config)
                self.vae.load_state_dict(converted_vae_checkpoint)
                # unet
                converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, self.unet.config)
                self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

                if is_lora:
                    self.pipeline = convert_lora(self.pipeline, state_dict, alpha=lora_alpha)

        self.pipeline.to("cuda")

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        print(f"sampling: {prompt} ...")
        outname = "output.gif"
        outpath = f"./{outname}"
        out_path = Path(tempfile.mkdtemp()) / "out.mp4"

        sample = self.pipeline(
            prompt,
            negative_prompt     = n_prompt,
            num_inference_steps = steps,
            guidance_scale      = guidance_scale,
            width               = 512,
            height              = 512,
            video_length        = 16,
        ).videos

        samples = torch.concat([sample])

        if not mp4:
            out_path = Path(tempfile.mkdtemp()) / "out.gif"
            save_videos_grid(samples, str(out_path) , n_rows=1)
        else:
            images = self.to_pil_images(sample, output_type="pil")
            out_dir = Path(tempfile.mkdtemp())
            out_path = out_dir / "out.mp4"
            for i, image in enumerate(images):
                image.save(str(out_dir / f"{i:03}.png"))
            os.system(f"ffmpeg -pattern_type glob -i '{str(out_dir)}/*.png' -movflags faststart -pix_fmt yuv420p -qp 17 "+ str(out_path))
        return Path(out_path)
