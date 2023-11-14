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
import torchvision.transforms as transforms

from einops import rearrange

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pretrained_model_path = "/AnimateDiff/models/StableDiffusion"
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        self.tokenizer_two = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")

    def to_pil_images(self, video_frames: torch.Tensor, output_type='pil'):
        to_pil = transforms.ToPILImage()
        video_frames = rearrange(video_frames, "b c f w h -> b f c w h")
        bsz = video_frames.shape[0]
        images = []
        for i in range(bsz):
            video = video_frames[i]
            for j in range(video.shape[0]):
                if output_type == "pil":
                    images.append(to_pil(video[j]))
                else:
                    images.append(video[j])
        return images

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
        prompt: str = Input(description="Input prompt", default="A panda standing on a surfboard in the ocean in sunset, 4k, high resolution. Realistic, Cinematic, high resolution"),
        n_prompt: str = Input(description="Negative prompt", default=""),
        steps: int = Input(description="Number of inference steps", ge=1, le=100, default=25),
        guidance_scale: float = Input(description="guidance scale", ge=1, le=10, default=8.5),
        seed: int = Input(description="Seed (0 = random, maximum: 2147483647)", ge=0, le=2147483647, default=None),
        mp4: bool = Input(description="Returns .mp4 if true or .gif if false", default=True)
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
        return out_path
