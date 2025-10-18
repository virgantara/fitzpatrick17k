import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from dataset import Fitzpatrick17kDataset
import gdown
import zipfile

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def download_dataset():

	# # !pip install gdown
	# import gdown
	# # url = 'https://drive.google.com/file/d/1AYMLQNb7cqNjSEXTFgqNTWthKxoRmkE9/view?usp=sharing'

	file_id = "1AYMLQNb7cqNjSEXTFgqNTWthKxoRmkE9"
	url = f"https://drive.google.com/uc?id={file_id}"

	output = "dataset.zip"  # or your file name
	gdown.download(url, output, quiet=False)

	with zipfile.ZipFile(output, 'r') as zip_ref:
	    zip_ref.extractall('dataset')  # Extract into 'dataset' folder



SEED = 42
set_seed(SEED)

csv_path = 'fitzpatrick17k.csv'
img_dir = 'dataset/data/finalfitz17k'
os.makedirs(img_dir, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),               # Resize slightly larger
    transforms.RandomResizedCrop(224),           # Random crop
    transforms.RandomHorizontalFlip(p=0.5),      # Flip left/right
    transforms.RandomRotation(degrees=15),       # Random rotation
    transforms.ToTensor(),
    transforms.Normalize((0.617, 0.477, 0.423), (0.232, 0.204, 0.207)),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),               # Deterministic resize
    transforms.ToTensor(),
    transforms.Normalize((0.617, 0.477, 0.423), (0.232, 0.204, 0.207)),
])

train_csv_path = "cv_splits/train_fold1.csv"
test_csv_path = "cv_splits/test_fold1.csv"

train_dataset = Fitzpatrick17kDataset(
    csv_file=train_csv_path,
    img_dir=img_dir,
    transform=train_transform,  # No transform yet
    img_ext=".jpg"
)

test_dataset = Fitzpatrick17kDataset(
    csv_file=test_csv_path,
    img_dir=img_dir,
    transform=test_transform,  # No transform yet
    img_ext=".jpg"
)

num_classes = train_dataset.num_classes
print("Total samples:", len(train_dataset))
print("Number of classes:", train_dataset.num_classes)
# print("Classes:", dataset.classes)
img, label = train_dataset[0]
print("Image shape:", img.shape)
print("Label index:", label, "->", train_dataset.idx2label[label])


# def calc_meanstd(dataset):
#     loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
#     # Step 3: compute mean and std
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     num_samples = 0
    
#     for images, _ in tqdm(loader, desc="Computing mean & std"):
#         batch_samples = images.size(0)  # number of images in batch
#         num_samples += batch_samples
    
#         # (B, C, H, W) -> sum over B,H,W
#         mean += images.sum(dim=[0, 2, 3])
#         std += (images ** 2).sum(dim=[0, 2, 3])
    
#     # Final mean and std
#     mean /= (num_samples * images.shape[2] * images.shape[3])
#     std = torch.sqrt(std / (num_samples * images.shape[2] * images.shape[3]) - mean ** 2)
    
#     print("Mean:", mean.tolist())
#     print("Std:", std.tolist())

# # calc_meanstd(dataset)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# BATCH_SIZE = 64
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# weights = models.ResNet18_Weights.IMAGENET1K_V1
# model = models.resnet18(weights=weights)
# # model.fc = nn.Linear(model.fc.in_features, num_classes)

# in_features = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Dropout(p=0.2),          # simple but effective
#     nn.Linear(in_features, num_classes)
# )

# model = model.to(device)

# for name, param in model.named_parameters():
#     if 'layer4' in name or 'fc' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False
# # for param in model.parameters():
# #     param.requires_grad = False
    
# # for param in model.fc.parameters():
# #     param.requires_grad = True

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-4)


# NUM_EPOCHS = 100

# best_acc = 0.0
# best_f1 = 0.0
# best_model_path = f"resnet18.pth"

# train_loss_list = []
# test_loss_list = []
# train_acc_list = []
# test_acc_list = []

# train_f1_list = []
# test_f1_list = []
# train_precision_list = []
# test_precision_list = []
# train_recall_list = []
# test_recall_list = []

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# for epoch in range(NUM_EPOCHS): 
#     model.train()
#     correct_train, total_train = 0, 0

#     running_train_loss = 0.0

#     all_train_preds = []
#     all_train_labels = []
#     for imgs, labels in tqdm(train_loader):
#         imgs, labels = imgs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_train_loss += loss.item() * imgs.size(0)

#         _, preds = torch.max(outputs, 1)
#         correct_train += (preds == labels).sum().item()
#         total_train += labels.size(0)

#         all_train_preds.extend(preds.cpu().numpy())
#         all_train_labels.extend(labels.cpu().numpy())

#     scheduler.step()

#     train_acc = 100 * correct_train / total_train
#     train_acc_list.append(train_acc)

#     train_loss = running_train_loss / total_train
#     train_loss_list.append(train_loss)

#     train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
#     train_recall = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
#     train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)

#     train_precision_list.append(train_precision)
#     train_recall_list.append(train_recall)
#     train_f1_list.append(train_f1)
    
#     model.eval()
#     correct_test, total_test = 0, 0
#     running_test_loss = 0.0

#     all_test_preds = []
#     all_test_labels = []
#     with torch.no_grad():
#         for imgs, labels in tqdm(test_loader):
#             imgs, labels = imgs.to(device), labels.to(device)
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             _, preds = torch.max(outputs, 1)
#             correct_test += (preds == labels).sum().item()

#             running_test_loss += loss.item() * imgs.size(0)
#             total_test += labels.size(0)

#             all_test_preds.extend(preds.cpu().numpy())
#             all_test_labels.extend(labels.cpu().numpy())

#     test_acc = 100 * correct_test / total_test
#     test_acc_list.append(test_acc)

#     test_precision = precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
#     test_recall = recall_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
#     test_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0)


#     test_loss = running_test_loss / total_test
#     test_loss_list.append(test_loss)

#     test_precision_list.append(test_precision)
#     test_recall_list.append(test_recall)
#     test_f1_list.append(test_f1)
#     if test_acc > best_acc:
#         best_acc = test_acc
#         torch.save(model.state_dict(), best_model_path)

#     if test_f1 > best_f1:
#         best_f1 = test_f1
#         torch.save(model.state_dict(), 'best_model_f1.pt')

#     print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
#           f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
#           f"Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f}")



# # Plot loss and accuracy curves
# plt.figure(figsize=(10, 4))

# # ===== LOSS =====
# plt.subplot(1, 2, 1)
# plt.plot(train_loss_list, label='Train Loss')
# plt.plot(test_loss_list, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss over Epochs')
# plt.legend()
# plt.grid(True)

# # ===== ACCURACY =====
# plt.subplot(1, 2, 2)
# plt.plot(train_acc_list, label='Train Accuracy')
# plt.plot(test_acc_list, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy over Epochs')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()