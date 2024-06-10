import os
import argparse
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Stable Diffusion Text-to-Image Inference.")
    parser.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Stable Diffusion model.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation.")
    parser.add_argument("--save_dir", type=str, default="/workspace/data/image",
                        help="Path to the save image directory.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate.")
    args = parser.parse_args()

    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to("cuda")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.num_images):
        # Process
        result = pipe(prompt=args.prompt).images[0]
        save_path = os.path.join(args.save_dir, f"generated_image_{i+1}.png")
        result.save(save_path)
        print(f"Image saved to {save_path}")
