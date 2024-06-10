import argparse
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

def generate_caption(image_path, processor, model, device, prompt):
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f'File not found: {image_path}')
        return None
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def main(image_path, model_name, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)

    caption = generate_caption(image_path, processor, model, device, prompt)
    if caption:
        print(f"Generated Caption: {caption}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BLIP2 model on an image.")
    parser.add_argument("--image_path", type=str, default='/workspace/data/images/xl_results.jpg', help="Path to the image file.")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-opt-2.7b", help="Name of the model to use.")
    parser.add_argument("--prompt", type=str, default='The photo of ', help="Prompt to use for the model.")

    args = parser.parse_args()
    main(args.image_path, args.model_name, args.prompt)
