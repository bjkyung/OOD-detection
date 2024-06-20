import os
import json
import argparse
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BertTokenizer, BertModel
from diffusers import AutoPipelineForImage2Image
from torchvision.transforms.functional import to_tensor
import lpips
import numpy as np
import safetensors.torch  # safetensors 패키지 로드
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CombinedDiscriminator(nn.Module):
    def __init__(self):
        super(CombinedDiscriminator, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last FC layer
        self.fc1 = nn.Linear(num_ftrs * 2 + 3, 256)  # Feature vectors of 2 images + 3 scores
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.sigmoid = nn.Sigmoid()

    def forward(self, img1, img2, scores):
        feat1 = self.resnet(img1)
        feat2 = self.resnet(img2)
        combined_feat = torch.cat((feat1, feat2, scores), dim=1)
        x = F.relu(self.fc1(combined_feat))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, annotation_path=None):
        self.image_dir = image_dir
        if annotation_path:
            with open(annotation_path, 'r') as f:
                self.annotations = json.load(f)['annotations']
            self.image_ids = [anno['image_id'] for anno in self.annotations if os.path.exists(os.path.join(image_dir, f"{str(anno['image_id']).zfill(12)}.jpg"))]
        else:
            self.image_ids = [int(os.path.splitext(filename)[0]) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_file = f"{str(image_id).zfill(12)}.jpg"
        image_path = os.path.join(self.image_dir, image_file)
        return image_id, image_path

def initialize_models():
    # Initialize BLIP2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained('/workspace/tools/infer/final_BLIP2')
    model = Blip2ForConditionalGeneration.from_pretrained('/workspace/tools/infer/final_BLIP2', torch_dtype=torch.float16)
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
    except AttributeError:
        print(f'AttributeError: {image_path}')
        return None
    inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Function to generate image using image-to-image
def generate_image(prompt, image_path, pipe, output_dir):
    try:
        image = Image.open(image_path).convert('RGB')
        result = pipe(prompt=prompt, image=image, disable_progress_bar=True).images[0]
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
    # Tokenize the captions
    inputs1 = tokenizer(caption1, return_tensors='pt', truncation=True, padding=True).to(device)
    inputs2 = tokenizer(caption2, return_tensors='pt', truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs1 = bert_model(**inputs1)
        outputs2 = bert_model(**inputs2)
    
    embedding1 = outputs1.last_hidden_state.mean(dim=1).cpu().numpy()
    embedding2 = outputs2.last_hidden_state.mean(dim=1).cpu().numpy()
    
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    normalized_similarity = (similarity + 1) / 2  # Normalize similarity to range [0, 1]
    
    # Tokenize the captions into words
    original_tokens = set(caption1.lower().split())
    new_tokens = set(caption2.lower().split())
    
    # Identify new tokens in the fake caption
    added_tokens = new_tokens - original_tokens
    
    return normalized_similarity, added_tokens  # Return normalized similarity and added tokens

def train(device, processor, model, tokenizer, bert_model, pipe, lpips_model, train_loader, val_loader, output_dir, epochs, val_annotations, discriminator, val_image_dir):
    optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    criterion = torch.nn.BCELoss()

    best_val_roc_auc = 0.0
    val_roc_auc = 0.0  # Initialize val_roc_auc with a default value
    for epoch in range(epochs):
        discriminator.train()
        running_loss = 0.0
        for image_id, image_path in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            image_path = image_path[0] if isinstance(image_path, tuple) else image_path
            original_image = Image.open(image_path).convert('RGB')
            original_image = to_tensor(original_image).unsqueeze(0).to(device)

            original_caption = generate_caption(image_path, processor, model, device)
            if original_caption is None:
                continue

            new_image_path = generate_image(original_caption, image_path, pipe, output_dir)
            if new_image_path is None:
                continue

            new_image = Image.open(new_image_path).convert('RGB')
            new_image = to_tensor(new_image).unsqueeze(0).to(device)

            new_caption = generate_caption(new_image_path, processor, model, device)
            if new_caption is None:
                continue

            caption_similarity, added_tokens = compute_caption_similarity(original_caption, new_caption, tokenizer, bert_model, device)
            caption_score = 1 - caption_similarity
            caption_score = max(0, caption_score)

            lpips_score = compute_lpips(image_path, new_image_path, lpips_model, device)
            if lpips_score is None:
                continue

            new_token_score = len(added_tokens)
            combined_scores = torch.tensor([caption_score, lpips_score, new_token_score], dtype=torch.float32).unsqueeze(0).to(device)

            label = torch.tensor([0.0], dtype=torch.float32).unsqueeze(0).to(device)  # Training with normal data only, label 0
            output = discriminator(original_image, new_image, combined_scores)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        if (epoch + 1) % 10 == 0:
            val_loss, val_roc_auc, val_f1 = validate(device, processor, model, tokenizer, bert_model, pipe, lpips_model, val_loader, discriminator, val_image_dir, output_dir, val_annotations)
            print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")
            print(f"Validation ROC-AUC after Epoch {epoch+1}: {val_roc_auc}")
            print(f"Validation F1 Score after Epoch {epoch+1}: {val_f1}")

            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                best_model_state = discriminator.state_dict()
                torch.save(best_model_state, os.path.join(output_dir, 'best_discriminator.pth'))
                print(f"Best discriminator saved with ROC-AUC {best_val_roc_auc}")

        scheduler.step(val_roc_auc)

        # Perform validation at the end of each epoch
        # val_loss, val_roc_auc, val_f1 = validate(device, processor, model, tokenizer, bert_model, pipe, lpips_model, val_loader, discriminator, val_image_dir, output_dir, val_annotations)
        # print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")
        # print(f"Validation ROC-AUC after Epoch {epoch+1}: {val_roc_auc}")
        # print(f"Validation F1 Score after Epoch {epoch+1}: {val_f1}")

        # # Update the best model
        # if val_roc_auc > best_val_roc_auc:
        #     best_val_roc_auc = val_roc_auc
        #     best_model_state = discriminator.state_dict()
        #     torch.save(best_model_state, os.path.join(output_dir, 'best_discriminator.pth'))
        #     print(f"Best discriminator saved with ROC-AUC {best_val_roc_auc}")

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch+1,
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss/len(train_loader),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

def validate(device, processor, model, tokenizer, bert_model, pipe, lpips_model, val_loader, discriminator, val_image_dir, output_dir, annotations):
    discriminator.eval()
    val_loss = 0.0
    criterion = torch.nn.BCELoss()
    all_labels = []
    all_outputs = []

    image_labels = {}
    for annotation in annotations['annotations']:
        image_id = str(annotation['image_id']).zfill(12)
        category_id = annotation['category_id']
        if image_id not in image_labels:
            image_labels[image_id] = 0
        if category_id >= 91:
            image_labels[image_id] = 1  # OOD

    print(f"Loaded {len(image_labels)} image labels for validation")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if isinstance(batch, torch.Tensor):
                image_ids = batch.numpy()
            elif isinstance(batch, (list, tuple)):
                image_ids = batch[0]
                if isinstance(image_ids, torch.Tensor):
                    image_ids = image_ids.numpy()
            else:
                image_ids = [batch]

            for image_id in image_ids:
                image_id = str(image_id).zfill(12)
                image_path = os.path.join(val_image_dir, f"{image_id}.jpg")

                if image_id not in image_labels:
                    print(f"Image ID {image_id} not found in image_labels")
                    continue

                original_image_path = image_path
                new_image_path = process_and_generate_images(original_image_path, processor, model, pipe, device, output_dir)
                if new_image_path is not None:
                    val_loss, output, label = process_images_and_evaluate(original_image_path, new_image_path, device, tokenizer, bert_model, lpips_model, discriminator, criterion, image_labels, val_loss, processor, model)
                    if output is not None:
                        all_outputs.append(output.item())
                        all_labels.append(label.item())

    if len(val_loader) > 0:
        val_loss /= len(val_loader)
    else:
        print("Warning: Validation loader is empty")

    if all_outputs:
        val_roc_auc = roc_auc_score(all_labels, all_outputs)
        optimal_threshold = find_optimal_threshold(all_labels, all_outputs)
        val_f1 = compute_f1_score(all_labels, all_outputs, optimal_threshold)
    else:
        val_roc_auc = 0.0
        val_f1 = 0.0

    print(f"Validation completed. Loss: {val_loss}, ROC-AUC: {val_roc_auc}, F1 Score: {val_f1}")

    return val_loss, val_roc_auc, val_f1

def process_and_generate_images(image_path, processor, model, pipe, device, output_dir):
    try:
        original_image = Image.open(image_path).convert('RGB')
        original_image = to_tensor(original_image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing original image {image_path}: {e}")
        return None

    original_caption = generate_caption(image_path, processor, model, device)
    if original_caption is None:
        print(f"Failed to generate caption for image: {image_path}")
        return None

    new_image_path = generate_image(original_caption, image_path, pipe, output_dir)
    if new_image_path is None:
        print(f"Failed to generate new image from caption for image: {image_path}")
        return None

    return new_image_path

def process_images_and_evaluate(original_image_path, new_image_path, device, tokenizer, bert_model, lpips_model, discriminator, criterion, image_labels, val_loss, processor, model):
    try:
        original_image = Image.open(original_image_path).convert('RGB')
        original_image = to_tensor(original_image).unsqueeze(0).to(device)

        new_image = Image.open(new_image_path).convert('RGB')
        new_image = to_tensor(new_image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing new image {new_image_path}: {e}")
        return val_loss, None, None

    original_caption = generate_caption(original_image_path, processor, model, device)
    if original_caption is None:
        print(f"Failed to generate caption for original image: {original_image_path}")
        return val_loss, None, None

    new_caption = generate_caption(new_image_path, processor, model, device)
    if new_caption is None:
        print(f"Failed to generate caption for new image: {new_image_path}")
        return val_loss, None, None

    caption_similarity, added_tokens = compute_caption_similarity(original_caption, new_caption, tokenizer, bert_model, device)
    caption_score = 1 - caption_similarity
    caption_score = max(0, caption_score)

    lpips_score = compute_lpips(original_image_path, new_image_path, lpips_model, device)
    if lpips_score is None:
        print(f"Failed to compute LPIPS score for images: {original_image_path}, {new_image_path}")
        return val_loss, None, None

    new_token_score = len(added_tokens)
    combined_scores = torch.tensor([caption_score, lpips_score, new_token_score], dtype=torch.float32).unsqueeze(0).to(device)

    image_id = os.path.splitext(os.path.basename(original_image_path))[0].zfill(12)
    label = torch.tensor([image_labels[image_id]], dtype=torch.float32).unsqueeze(0).to(device)
    output = discriminator(original_image, new_image, combined_scores)

    if output is not None:
        print(f"Output: {output.item()} | Label: {label.item()}")
    else:
        print(f"Discriminator output is None for images: {original_image_path}, {new_image_path}")
        return val_loss, None, None

    loss = criterion(output, label)
    print(f"Intermediate Loss: {loss.item()}")
    val_loss += loss.item()

    return val_loss, output, label

def find_optimal_threshold(labels, scores):
    thresholds = sorted(set(scores))
    best_threshold = thresholds[0]
    best_f1 = 0

    for threshold in thresholds:
        predictions = [1 if score >= threshold else 0 for score in scores]
        current_f1 = f1_score(labels, predictions)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold

def compute_f1_score(labels, scores, threshold):
    predictions = [1 if score >= threshold else 0 for score in scores]
    return f1_score(labels, predictions)

def main(train_image_dir, val_image_dir, val_annotation_path, output_dir, lora_weights_path, epochs, batch_size):
    device, processor, model, tokenizer, bert_model, pipe, lpips_model = initialize_models()
    lora_weights = safetensors.torch.load_file(lora_weights_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize datasets and dataloaders
    train_dataset = ImageCaptionDataset(train_image_dir)  # No annotation needed for training dataset
    val_dataset = ImageCaptionDataset(val_image_dir, val_annotation_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Set batch_size to 1 for validation

    # Load validation annotations
    with open(val_annotation_path, 'r') as f:
        val_annotations = json.load(f)

    # Initialize discriminator
    discriminator = CombinedDiscriminator().to(device)

    train(device, processor, model, tokenizer, bert_model, pipe, lpips_model, train_loader, val_loader, output_dir, epochs, val_annotations, discriminator, val_image_dir)

    # Perform final validation
    val_loss, val_roc_auc, val_f1 = validate(device, processor, model, tokenizer, bert_model, pipe, lpips_model, val_loader, discriminator, val_image_dir, output_dir, val_annotations)
    print(f"Final Validation Loss: {val_loss}")
    print(f"Final Validation ROC-AUC: {val_roc_auc}")
    print(f"Final Validation F1 Score: {val_f1}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate OOD detection on images.")
    parser.add_argument("--train_image_dir", type=str, required=True, help="Directory containing training images.")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Directory containing validation images.")
    parser.add_argument("--val_annotation_path", type=str, required=True, help="Path to the validation annotation file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images and checkpoints.")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="Path to the LoRA weights.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")

    args = parser.parse_args()
    main(args.train_image_dir, args.val_image_dir, args.val_annotation_path, args.output_dir, args.lora_weights_path, args.epochs, args.batch_size)
