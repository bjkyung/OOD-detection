import argparse
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model
import torch

def main(image_url, model_name, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)

    image = Image.open(requests.get(image_url, stream=True).raw)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    outputs = model(**inputs)
    print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BLIP2 model on an image.")
    parser.add_argument("--image_url", type=str, default='/workspace/data/images/xl_results.jpg', help="URL of the image.")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-opt-2.7b", help="Name of the model to use.")
    parser.add_argument("--prompt", type=str, default='The photo of ', help="Prompt to use for the model.")

    args = parser.parse_args()
    main(args.image_url, args.model_name, args.prompt)
