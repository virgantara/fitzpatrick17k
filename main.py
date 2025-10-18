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
import argparse
import wandb
from modelku import get_model

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


def train(args):
	wandb.init(project="SkinDisease", name=args.exp_name)
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

	fold = args.fold
	train_csv_path = "cv_splits/train_fold"+fold+".csv"
	test_csv_path = "cv_splits/test_fold"+fold+".csv"

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
	# print("Total samples:", len(train_dataset))
	# print("Number of classes:", train_dataset.num_classes)
	# # print("Classes:", dataset.classes)
	# img, label = train_dataset[0]
	# print("Image shape:", img.shape)
	# print("Label index:", label, "->", train_dataset.idx2label[label])

	args.num_classes = num_classes

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
	test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)

	model = get_model(args)

	model = model.to(device)

	wandb.watch(model)

	total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)


	NUM_EPOCHS = args.epochs

	best_acc = 0.0
	best_f1 = 0.0
	best_model_path = f"resnet18.pth"

	train_loss_list = []
	test_loss_list = []
	train_acc_list = []
	test_acc_list = []

	train_f1_list = []
	test_f1_list = []
	train_precision_list = []
	test_precision_list = []
	train_recall_list = []
	test_recall_list = []

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
	for epoch in range(NUM_EPOCHS): 
		wandb_log = {}

	    model.train()
	    correct_train, total_train = 0, 0

	    running_train_loss = 0.0

	    all_train_preds = []
	    all_train_labels = []
	    for imgs, labels in tqdm(train_loader):
	        imgs, labels = imgs.to(device), labels.to(device)

	        optimizer.zero_grad()
	        outputs = model(imgs)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()
	        
	        running_train_loss += loss.item() * imgs.size(0)

	        _, preds = torch.max(outputs, 1)
	        correct_train += (preds == labels).sum().item()
	        total_train += labels.size(0)

	        all_train_preds.extend(preds.cpu().numpy())
	        all_train_labels.extend(labels.cpu().numpy())

	    scheduler.step()

	    train_acc = 100 * correct_train / total_train
	    train_acc_list.append(train_acc)

	    train_loss = running_train_loss / total_train
	    train_loss_list.append(train_loss)

	    train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
	    train_recall = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
	    train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)

	    train_precision_list.append(train_precision)
	    train_recall_list.append(train_recall)
	    train_f1_list.append(train_f1)

	    wandb_log['Train Loss'] = train_loss
        wandb_log['Train Precision'] = train_precision
        wandb_log['Train F1'] = train_f1
        wandb_log['Train Recall'] = train_recall
        wandb_log['Train Acc'] = train_acc
	    
	    model.eval()
	    correct_test, total_test = 0, 0
	    running_test_loss = 0.0

	    all_test_preds = []
	    all_test_labels = []
	    with torch.no_grad():
	        for imgs, labels in tqdm(test_loader):
	            imgs, labels = imgs.to(device), labels.to(device)
	            outputs = model(imgs)
	            loss = criterion(outputs, labels)
	            _, preds = torch.max(outputs, 1)
	            correct_test += (preds == labels).sum().item()

	            running_test_loss += loss.item() * imgs.size(0)
	            total_test += labels.size(0)

	            all_test_preds.extend(preds.cpu().numpy())
	            all_test_labels.extend(labels.cpu().numpy())

	    test_acc = 100 * correct_test / total_test
	    test_acc_list.append(test_acc)

	    test_precision = precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
	    test_recall = recall_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
	    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0)


	    test_loss = running_test_loss / total_test
	    test_loss_list.append(test_loss)

	    test_precision_list.append(test_precision)
	    test_recall_list.append(test_recall)
	    test_f1_list.append(test_f1)
	    if test_acc > best_acc:
	        best_acc = test_acc
	        torch.save(model.state_dict(), best_model_path)

	    if test_f1 > best_f1:
	        best_f1 = test_f1
	        torch.save(model.state_dict(), 'best_model_f1.pt')

	    wandb_log['Test Loss'] = test_loss
        wandb_log['Test Precision'] = test_precision
        wandb_log['Test F1'] = test_f1
        wandb_log['Test Recall'] = test_recall
        wandb_log['Test Acc'] = test_acc
        wandb.log(wandb_log)
	    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
	          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
	          f"Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f}")



	# Plot loss and accuracy curves
	plt.figure(figsize=(10, 4))

	# ===== LOSS =====
	plt.subplot(1, 2, 1)
	plt.plot(train_loss_list, label='Train Loss')
	plt.plot(test_loss_list, label='Test Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss over Epochs')
	plt.legend()
	plt.grid(True)

	# ===== ACCURACY =====
	plt.subplot(1, 2, 2)
	plt.plot(train_acc_list, label='Train Accuracy')
	plt.plot(test_acc_list, label='Test Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (%)')
	plt.title('Accuracy over Epochs')
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model_name', type=str, default='van', metavar='N',
                        choices=['resnet18','van'])
    parser.add_argument('--attn_type', type=str, default='SA', metavar='N',
                        choices=['SA','LKA','LSKA','CBAM'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
	train(args)