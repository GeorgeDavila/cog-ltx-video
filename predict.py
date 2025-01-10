# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import sys
import torch
import shutil
from PIL import Image
from typing import List
from diffusers import StableDiffusionXLPipeline
import math

base_model_path = "Lightricks/LTX-Video"
#LoRA_PATH = "hfpath/sdxl-beethoven-spectrograms"
#LoRA_file = "lora.safetensors"
device = "cuda"
MODEL_CACHE = "model-cache"

import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

def load_image(path): #for img2img
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # load SDXL pipeline
        self.pipe = LTXPipeline.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            watermark=None,
            safety_checker=None,
            #variant="fp16",
        ).to(device)

        #self.pipe.load_lora_weights(LoRA_PATH, weight_name=LoRA_file)

    def predict(
        self,
        myprompt: str = Input(
            description="Input prompt",
            default="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
        ),
        #promptAddendum: str = Input(
        #    description="Extra terms to add to end of prompt",
        #    default="",
        #),
        negative_prompt: str = Input(
            description="Negative Prompt",
            default="watermark, text, deformed, worst quality, inconsistent motion, blurry, jittery, distorted"
        ),
        outWidth: int = Input(
            description="width of output",
            ge=128,
            le=2048,
            default=768, # 640
        ),
        outHeight: int = Input(
            description="height of output",
            ge=128,
            le=2048,
            default=512, # 480
        ),
        guidanceScale: float = Input(
            description="Guidance scale (influence of input text on generation)",
            ge=1.0,
            le=10.0,
            default=3.0
        ),
        num_frames: int = Input(
            description="Number of images to output.",
            default=97,
            choices=[97, 129, 161, 193, 225, 257],
        ),
        num_outputs: int = Input(
            description="Number of outputs.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        outFPS: int = Input(
            description="Output FPS",
            default=24,
        ),
        decodeTimestepParam: float = Input(
            description="decodeTimestepParam",
            ge=0.005,
            le=1.0,
            default=0.030
        ),
        decodeNoiseScaleParam: float = Input(
            description="decodeNoiseScaleParam",
            ge=0.0005,
            le=1.0,
            default=0.0250
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        while (myprompt[-1] == " ") or (myprompt[-1] == "\n"): #remove user whitespaces
            myprompt = myprompt[:-1]
        myprompt = myprompt #+ promptAddendum
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if outWidth % 32 != 0:
            outWidth = math.floor(outWidth/32)*32
            print(f"WARNING: outWidth must be a multiple of 32, resizing outWidth to {outWidth}")
        if outHeight % 32 != 0:
            outHeight = math.floor(outHeight/32)*32
            print(f"WARNING: outHeight must be a multiple of 32, resizing outHeight to {outHeight}")

        #video = self.pipe(
        #    prompt=myprompt,
        #    negative_prompt=negative_prompt,
        #    width=outWidth,
        #    height=outHeight,
        #    num_frames=num_frames,
        #    decode_timestep=0.03,
        #    decode_noise_scale=0.025,
        #    num_inference_steps=num_inference_steps,
        #    guidance_scale=guidanceScale,
        #    num_videos_per_prompt=num_outputs,
        #).frames[0]

        videosObj = self.pipe(
            prompt=myprompt,
            negative_prompt=negative_prompt,
            width=outWidth,
            height=outHeight,
            num_frames=num_frames,
            decode_timestep=decodeTimestepParam,
            decode_noise_scale=decodeNoiseScaleParam,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidanceScale,
            num_videos_per_prompt=num_outputs,
        ).frames

        #output_filename = "/tmp/output.mp4"
        #export_to_video(video, output_filename, fps=outFPS)
        
        output_paths = []
        for i, _ in enumerate(videosObj):
            output_path = f"/tmp/out-{i}.mp4"
            export_to_video(videosObj[i], output_path, fps=outFPS)
            output_paths.append(Path(output_path))
            
        return output_paths
