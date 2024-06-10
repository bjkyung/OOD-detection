import os
import torch
import argparse
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image

def load_pipeline():
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    return pipeline, pipeline_text2image

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion XL Text-to-Image Inference.")
    parser.add_argument("--prompt", type=str, default='A man skiing downhill', help="Text prompt to generate image.")
    parser.add_argument("--save_dir", type=str, default='/workspace/data/images', help="Path to save the generated images.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate.")
    args = parser.parse_args()

    pipeline, _ = load_pipeline()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.num_images):
        image = pipeline(prompt=args.prompt).images[0]
        save_path = os.path.join(args.save_dir, f"xl_results_{i+1}.jpg")
        image.save(save_path)
        print(f"Image saved to {save_path}")

if __name__ == "__main__":
    main()
