# src/Unlabeled.py
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class UnlabeledImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        # ดึงเฉพาะไฟล์ภาพ
        self.image_files = [
            f for f in self.root.iterdir()
            if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')  # แปลงเป็น RGB เพื่อความปลอดภัย
        if self.transform:
            image = self.transform(image)
        # ส่งคืนทั้ง tensor ของภาพ และชื่อไฟล์เต็ม (สำหรับการคัดลอก)
        return image, str(img_path)