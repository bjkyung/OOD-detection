import os
import json
import argparse
from PIL import Image, UnidentifiedImageError
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BertTokenizer, BertModel
from diffusers import AutoPipelineForImage2Image
from torchvision.transforms.functional import to_tensor
import lpips
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import safetensors.torch  # safetensors 패키지 로드

def initialize_models():
    # Initialize BLIP2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16)
    model.to(device)

    # Initialize BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)

    # Initialize Stable Diffusion
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16,
                                                      variant="fp16", use_safetensors=True)
    pipe.to(device)

    # LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)

    return device, processor, model, tokenizer, bert_model, pipe, lpips_model

# Function to generate caption
def generate_caption(image_path, processor, model, device):
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f'File not found: {image_path}')
        return None
    inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Function to generate image using image-to-image
def generate_image(prompt, image_path, pipe, output_dir):
    try:
        image = Image.open(image_path)
        result = pipe(prompt=prompt, image=image).images[0]
        save_path = os.path.join(output_dir, os.path.basename(image_path))
        result.save(save_path)
        return save_path
    except UnidentifiedImageError:
        print(f'이미지를 식별할 수 없습니다: {image_path}')
        return None
    except FileNotFoundError:
        print(f'File not found: {image_path}')
        return None

# Function to compute LPIPS
def compute_lpips(img1_path, img2_path, lpips_model, device):
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
    except FileNotFoundError:
        print(f'File not found: {img1_path} or {img2_path}')
        return None

    # Resize images to the same size
    if img1.size != img2.size:
        min_size = (min(img1.size[0], img2.size[0]), min(img1.size[1], img2.size[1]))
        img1 = img1.resize(min_size, Image.BICUBIC)
        img2 = img2.resize(min_size, Image.BICUBIC)

    img1 = to_tensor(img1).unsqueeze(0).to(device)
    img2 = to_tensor(img2).unsqueeze(0).to(device)
    return lpips_model(img1, img2).item()

# Function to compute cosine similarity between two captions using BERT embeddings
def compute_caption_similarity(caption1, caption2, tokenizer, bert_model, device):
    inputs1 = tokenizer(caption1, return_tensors='pt', truncation=True, padding=True).to(device)
    inputs2 = tokenizer(caption2, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs1 = bert_model(**inputs1)
        outputs2 = bert_model(**inputs2)
    embedding1 = outputs1.last_hidden_state.mean(dim=1).cpu().numpy()
    embedding2 = outputs2.last_hidden_state.mean(dim=1).cpu().numpy()
    return cosine_similarity(embedding1, embedding2)[0][0]

def main(image_dir, annotation_path, output_dir, lora_weights_path, threshold=0.5):
    device, processor, model, tokenizer, bert_model, pipe, lpips_model = initialize_models()
    lora_weights = safetensors.torch.load_file(lora_weights_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load annotations
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Collect image-wise labels
    image_labels = {}
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        if image_id not in image_labels:
            image_labels[image_id] = 0
        if category_id >= 91:
            image_labels[image_id] = 1

    # Process images
    ood_scores = {}
    for image_id in image_labels.keys():
        image_file = f"{str(image_id).zfill(12)}.jpg"
        image_path = os.path.join(image_dir, image_file)

        # Generate original caption
        original_caption = generate_caption(image_path, processor, model, device)
        if original_caption is None:
            continue

        # Generate new image
        new_image_path = generate_image(original_caption, image_path, pipe, output_dir)
        if new_image_path is None:
            continue

        # Generate new caption
        new_caption = generate_caption(new_image_path, processor, model, device)
        if new_caption is None:
            continue

        # Compute caption similarity
        caption_similarity = compute_caption_similarity(original_caption, new_caption, tokenizer, bert_model, device)
        caption_score = 1 - caption_similarity  # Higher score means less similar

        # Compute LPIPS
        lpips_score = compute_lpips(image_path, new_image_path, lpips_model, device)
        if lpips_score is None:
            continue

        # Calculate final score (customizable)
        final_score = caption_score + lpips_score

        ood_scores[image_id] = final_score

    # Prepare data for ROC-AUC calculation
    scores = [ood_scores[image_id] for image_id in image_labels.keys() if image_id in ood_scores]
    labels = [image_labels[image_id] for image_id in image_labels.keys() if image_id in ood_scores]

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(labels, scores)
    print(f"ROC-AUC: {roc_auc}")

    # Determine OOD based on threshold
    ood_results = {image_id: score for image_id, score in ood_scores.items() if score >= threshold}

    # Print results
    print("OOD Results:", ood_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OOD detection on images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to the annotation file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="Path to the LoRA weights.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for determining OOD.")

    args = parser.parse_args()
    main(args.image_dir, args.annotation_path, args.output_dir, args.lora_weights_path, args.threshold)
