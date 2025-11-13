import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# =====================================
# 1️⃣ เลือกโมเดล (เปลี่ยนชื่อไฟล์ checkpoint ตามจริง)
# =====================================
model_name = "resnet"  # หรือ "resnet" / "efficientnet"
num_classes = 9  # จำนวนคลาสของคุณ

if model_name == "resnet":
    model = models.resnet18(weights=None)
    target_layer = model.layer4[-1]
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    ckpt_path = "Output/checkpoints/resnet_pretrain.pth"

elif model_name == "efficientnet":
    model = models.efficientnet_b0(weights=None)
    target_layer = model.features[-1]
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    ckpt_path = "./Output/checkpoints/efficientnet_b0.pth"

elif model_name == "convnext":
    model = models.convnext_tiny(weights=None)
    target_layer = model.features[-1]
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    ckpt_path = "./Output/checkpoints/convnext_tiny.pth"

else:
    raise ValueError("❌ Unsupported model")

# โหลด weights ที่เทรนไว้
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

class_names = ["center", "donut", "edge_ring", "loc", "edge_loc", "scratch","random", "near_full"]

# =====================================
# 2️⃣ โหลดภาพ wafer
# =====================================
img_path = r"C:\Users\User\Documents\GitHub\testwork\little_wafer\Data\pre_train\center\center_20251108_193939_001_clean.png"
img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)

preds = model(input_tensor)
# =====================================
# 3️⃣ ใช้ Grad-CAM
# =====================================
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

# ถ้าอยากดู class ที่โมเดลทำนายเอง
with torch.no_grad():
    preds = model(input_tensor)
    pred_class = preds.argmax(dim=1).item()
    predicted_class_name = class_names[pred_class]
    
print(f"🔹 Predicted class index: {pred_class}({predicted_class_name}) ")

# สร้าง heatmap สำหรับคลาสที่โมเดลทำนาย
targets = [ClassifierOutputTarget(pred_class)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

# =====================================
# 4️⃣ แปลงกลับเป็นภาพ + overlay
# =====================================
rgb_img = np.array(img.resize((224, 224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title("Original Wafer")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title(f"Grad-CAM Focus ({model_name.upper()})")
plt.axis('off')
plt.show()
