# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
sys.path.extend(['/AnimateDiff'])
import tempfile
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from omegaconf import OmegaConf
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, load_weights
from diffusers.utils.import_utils import is_xformers_available

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pretrained_model_path = "/AnimateDiff/models/StableDiffusion"
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        self.tokenizer_two = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")

    def predict(
        self,
        motion_module: str = Input(
            description="Select a Motion Model (currently only one available)",
            default="mm_sdxl_v10_beta",
            choices=[
                "mm_sdxl_v10_beta"
            ],
        ),
        checkpoint: str = Input(
            default="dynavision",
            choices=[
                "dynavision",
                "dreamshaper",
                "deepblue"
            ],
            description="Select a model checkpoint",
        ),
        use_checkpoint: bool = Input(default=False),
        aspect: str = Input(
            default="1:1",
            choices=[
                "9:16",
                "2:3",
                "1:1",
                "3:2",
                "16:9",
            ],
            description="Aspect ratio"
        ),
        video_length: int = Input(description="Video length", ge=16, default=16),
        prompt: str = Input(description="Input prompt", default="A panda standing on a surfboard in the ocean in sunset, 4k, high resolution.Realistic, Cinematic, high resolution"),
        n_prompt: str = Input(description="Negative prompt", default=""),
        steps: int = Input(description="Number of inference steps", ge=1, le=100, default=25),
        guidance_scale: float = Input(description="guidance scale", ge=1, le=10, default=8.5),
        seed: int = Input(description="Seed (0 = random, maximum: 2147483647)", ge=0, le=2147483647, default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        base=""

        aspect_to_width_height = {
            "9:16": (768,1344),
            "2:3": (832,1216),
            "1:1": (1024,1024),
            "3:2": (1216,832),
            "16:9": (1344,768),
        }
        (width, height) = aspect_to_width_height[aspect]
        # Create paths and load motion model
        newPath = f"/AnimateDiff/models/DreamBooth_LoRA/{checkpoint}.safetensors"
        motion_path = "/AnimateDiff/models/Motion_Module/"+motion_module+".ckpt"
        inference_config_file = "/AnimateDiff/configs/inference/inference.yaml"
        # Load configuration
        inference_config = OmegaConf.load(inference_config_file)

        self.unet = UNet3DConditionModel.from_pretrained_2d(
            "/AnimateDiff/models/StableDiffusion",
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
        )
        scheduler = EulerDiscreteScheduler(timestep_spacing='leading', steps_offset=1,**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
        self.pipeline = AnimationPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=scheduler,
            text_encoder_2 = self.text_encoder_two, tokenizer_2=self.tokenizer_two
        )
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        
        # TODO: Check for LoRA
        # Load model weights
        if use_checkpoint:
            self.pipeline = load_weights(
                pipeline = self.pipeline,
                motion_module_path = motion_path,
                ckpt_path = newPath,
                lora_path = "",
                lora_alpha = 0.8
            )
        else:
            self.pipeline = load_weights(
                pipeline = self.pipeline,
                motion_module_path = motion_path,
                ckpt_path = "",
                lora_path = "",
                lora_alpha = 0.8
            )

        self.pipeline.unet = self.pipeline.unet.half()
        self.pipeline.text_encoder = self.pipeline.text_encoder.half()
        self.pipeline.text_encoder_2 = self.pipeline.text_encoder_2.half()
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()

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
            width               = width,
            height              = height,
            single_model_length = video_length,
        ).videos

        samples = torch.concat([sample])
        save_videos_grid(samples, outpath , n_rows=1)
        os.system("ffmpeg -i output.gif -movflags faststart -pix_fmt yuv420p -qp 17 "+ str(out_path))
        # Fix so that it returns the actual gif or mp4 in replicate
        print(f"saved to file")
        return Path(out_path)
