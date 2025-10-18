from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os

class Fitzpatrick17kDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None, img_ext=".jpg"):
        """
        Args:
            csv_file (str): Path to the CSV file (fitzpatrick17k.csv)
            img_dir (str): Path to the folder containing images
            transform (callable, optional): Transform to apply to the images
            target_transform (callable, optional): Transform to apply to labels
            img_ext (str): Image file extension, e.g., '.jpg' or '.png'
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_ext = img_ext
        self.classes = sorted(self.data['label'].unique())

        self.num_classes = len(self.classes)

        # Create a mapping from label string to integer
        self.label2idx = {label: idx for idx, label in enumerate(self.classes)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # Precompute image paths and labels with tqdm
        self.samples = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Loading dataset"):
            img_name = f"{row['md5hash']}{self.img_ext}"
            img_path = os.path.join(self.img_dir, img_name)
            label = self.label2idx[row['label']]
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label