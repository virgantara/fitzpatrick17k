import os
import random
import zipfile
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb
import gdown
from dataset import Fitzpatrick17kDataset
from modelku import get_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_dataset():
    file_id = "1AYMLQNb7cqNjSEXTFgqNTWthKxoRmkE9"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "dataset.zip"

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

    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.617, 0.477, 0.423),
                             (0.232, 0.204, 0.207)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.617, 0.477, 0.423),
                             (0.232, 0.204, 0.207)),
    ])

    # --- Dataset ---
    fold = str(args.fold)
    train_csv_path = f"cv_splits/train_fold{fold}.csv"
    test_csv_path = f"cv_splits/test_fold{fold}.csv"

    train_dataset = Fitzpatrick17kDataset(
        csv_file=train_csv_path,
        img_dir=img_dir,
        transform=train_transform,
        img_ext=".jpg"
    )

    test_dataset = Fitzpatrick17kDataset(
        csv_file=test_csv_path,
        img_dir=img_dir,
        transform=test_transform,
        img_ext=".jpg"
    )

    num_classes = train_dataset.num_classes
    args.num_classes = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=8)

    model = get_model(args)

    # --- Load pretrained SimCLR model ---
    # if args.simclr_ckpt and os.path.exists(args.simclr_ckpt):
    #     print(f"Loading pretrained SimCLR weights from {args.simclr_ckpt}")
    #     ckpt = torch.load(args.simclr_ckpt, map_location=device)

    #     # Try to load only matching keys (in case projection head differs)
    #     state_dict = ckpt.get('state_dict', ckpt)
    #     new_state_dict = {}

    #     for k, v in state_dict.items():
    #         # Common SimCLR formats
    #         for prefix in [
    #             "backbone.",          # your ResNetSimCLR backbone
    #             "encoder.",           # some SimCLR implementations
    #             "model.backbone.",    # some wrappers
    #             "module.backbone.",   # DDP variant
    #             "module.encoder_q.",  # MoCo/SimCLR variant
    #             "module."             # plain DDP
    #         ]:
    #             if k.startswith(prefix):
    #                 k = k[len(prefix):]

    #         # skip the MLP projection head
    #         if k.startswith("fc.") or "projection" in k:
    #             continue

    #         new_state_dict[k] = v

    #     model_dict = model.state_dict()

    #     pretrained_dict = {
    #         k: v for k, v in new_state_dict.items()
    #         if k in model_dict and model_dict[k].shape == v.shape
    #     }

    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict, strict=False)
    #     print(f"Loaded {len(pretrained_dict)} matching layers from SimCLR checkpoint.")
    # else:
    #     print("No SimCLR checkpoint provided or file not found. Training from scratch.")

    # # --- Replace projection head with classifier ---
    # if hasattr(model, 'fc'):  # for resnet-based backbones
    #     in_features = model.fc.in_features
    #     model.fc = nn.Sequential(
    #         nn.Dropout(p=0.2),
    #         nn.Linear(in_features, args.num_classes)
    #     )
    # elif hasattr(model, 'head'):  # for VAN or ViT-style models
    #     in_features = model.head.in_features
    #     model.head = nn.Linear(in_features, args.num_classes)

    # --- Optionally freeze lower layers ---
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'head' not in name:
                param.requires_grad = False
        print("Backbone frozen, training only classifier head.")

    model = model.to(device)
    wandb.watch(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    NUM_EPOCHS = args.epochs
    best_acc = 0.0
    best_f1 = 0.0
    best_model_path = "resnet18.pth"

    # --- Metric logs ---
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    train_f1_list, test_f1_list = [], []
    train_precision_list, test_precision_list = [], []
    train_recall_list, test_recall_list = [], []

    for epoch in range(NUM_EPOCHS):
        wandb_log = {}
        model.train()

        correct_train, total_train = 0, 0
        running_train_loss = 0.0
        all_train_preds, all_train_labels = [], []

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
        train_loss = running_train_loss / total_train
        train_precision = precision_score(all_train_labels, all_train_preds,
                                          average='weighted', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds,
                                    average='weighted', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds,
                            average='weighted', zero_division=0)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        train_precision_list.append(train_precision)
        train_recall_list.append(train_recall)
        train_f1_list.append(train_f1)

        wandb_log.update({
            'Train Loss': train_loss,
            # 'Train Precision': train_precision,
            # 'Train Recall': train_recall,
            'Train F1': train_f1,
            'Train Acc': (correct_train / total_train)
        })

        # --- Evaluation ---
        model.eval()
        correct_test, total_test = 0, 0
        running_test_loss = 0.0
        all_test_preds, all_test_labels = [], []

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
        test_loss = running_test_loss / total_test
        test_precision = precision_score(all_test_labels, all_test_preds,
                                         average='weighted', zero_division=0)
        test_recall = recall_score(all_test_labels, all_test_preds,
                                   average='weighted', zero_division=0)
        test_f1 = f1_score(all_test_labels, all_test_preds,
                           average='macro', zero_division=0)

        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        test_precision_list.append(test_precision)
        test_recall_list.append(test_recall)
        test_f1_list.append(test_f1)

        # --- Save best models ---
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), 'best_model_f1.pt')

        wandb_log.update({
            'Test Loss': test_loss,
            # 'Test Precision': test_precision,
            # 'Test Recall': test_recall,
            'Test F1': test_f1,
            'Test Acc': (correct_test / total_test)
        })

        wandb.log(wandb_log)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f}")

    # --- Plot curves ---
    # plt.figure(figsize=(10, 4))

    # # Loss
    # plt.subplot(1, 2, 1)
    # plt.plot(train_loss_list, label='Train Loss')
    # plt.plot(test_loss_list, label='Test Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss over Epochs')
    # plt.legend()
    # plt.grid(True)

    # # Accuracy
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skin Disease Classification')

    parser.add_argument('--exp_name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--model_name', type=str, default='van',
                        choices=['resnet18', 'van'])
    parser.add_argument('--van_arch', type=str, default='van_b0',
                        choices=['van_b0', 'van_b1','van_b2','van_b3'])
    parser.add_argument('--attn_type', type=str, default='SA',
                        choices=['SA', 'LKA', 'LSKA', 'CBAM'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--simclr_ckpt', type=str, default=None,
                        help='Path to pretrained SimCLR checkpoint (.pth)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone and train only classification head')

    args = parser.parse_args()
    train(args)
