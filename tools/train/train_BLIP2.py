import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from torchvision import transforms
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence

# Define the custom dataset
class CustomCaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.captions[idx]['image']
        img_path = f"{self.image_dir}/{img_name}"
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]['caption']
        return image, caption

def collate_fn(batch):
    images, captions = zip(*batch)
    pixel_values = torch.stack(images)
    text_inputs = processor.tokenizer(captions, padding=True, return_tensors="pt")
    
    return {
        "pixel_values": pixel_values,
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "labels": text_inputs["input_ids"],
    }

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

image_dir = "/nfs_shared/JK/OOD_coco2017/train2017/skis_35/images"
captions_file = "/nfs_shared/JK/OOD_coco2017/train2017/skis_35/train_captions.json"

dataset = CustomCaptionDataset(image_dir, captions_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Load processor
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Configure and load model with quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "ybelkada/blip2-opt-2.7b-fp16-sharded",
    device_map={"": 1},  # Explicitly setting model to use GPU 0
    quantization_config=quantization_config
)

# Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)
model = get_peft_model(model, config)

# Ensure all model parameters are on the same device
device = torch.device("cuda:1")
model.to(device)

# Print trainable parameters
model.print_trainable_parameters()

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-4)
num_epochs = 200
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Ensure the model and optimizer are on the correct device
model.to(device)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

model.train()

# Directory for saving checkpoints
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    print("Epoch:", epoch)
    for idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        progress_bar.update(1)

    # Save checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

print("Training completed!")

# Save the final model
model.save_pretrained("./final_model")
processor.save_pretrained("./final_model")

# Save the final model
model.save_pretrained("./final_model")
processor.save_pretrained("./final_model")
