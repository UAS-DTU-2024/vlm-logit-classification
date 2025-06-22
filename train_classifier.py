import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataloader import VisionFineTuneDataset, custom_collate_fxn
from classification_model import VLM_Embed_Classifier
from tokens_retrieve import *

img_dir = ""
output_dir = ""
gt_csv = ""
device = "cuda:0"
num_epochs = 20

dataset = VisionFineTuneDataset(img_dir, gt_csv)
train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(
    train_dataset,
    batch_size = 4,
    shuffle = True,
    collate_fn = custom_collate_fxn
)

val_loader = DataLoader(
    val_dataset,
    batch_size = 4,
    shuffle = True,
    collate_fn = custom_collate_fxn
)

model = VLM_Embed_Classifier(dim = 4096, num_heads =4, num_blocks=4, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)

val_acc = 0.0

for epoch in range(0, num_epochs):
    model.train()
    loss = 0.0
    correct = 0
    total = 0

    for images, labels, indices in train_loader:
        for image in images:
            hidden_states, prompt = process_images(image, img_dir, output_dir, device = device, quantize_type="fp16")
        filtered_states = filter_embeddings(hidden_states, indices, prompt)
        cls_embeds = prepend_cls_token(filtered_states)
        cls_embeds = cls_embeds.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(cls_embeds)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, indices in train_loader:
            for image in images:
                hidden_states, prompt = process_images(image, img_dir, output_dir, device = device, quantize_type="fp16")
            filtered_states = filter_embeddings(hidden_states, indices, prompt)
            cls_embeds = prepend_cls_token(filtered_states)
            cls_embeds = cls_embeds.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(cls_embeds)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    print(f"Epoch {epoch} - "
          f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} - "
          f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} - ")