import os
import yaml
from pathlib import Path
from PIL import Image
import numpy as np

# Load config
CONFIG_PATH = 'configs/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def resize_img(input_folder=None, output_folder=None, target_size=(224,224)):
    """ Resize all images in the input folder to the target size. """
    if input_folder is None:
        input_folder = config['data']['wafer_map_png']  # ใช้ default จาก config
    if output_folder is None:
        output_folder = config['data']['wafer_map_resized']
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    supported_extensions = ('.png', '.jpg', '.jpeg')
    resized_files = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    # Resize
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    # Save
                    output_path = os.path.join(output_folder, filename)
                    img_resized.save(output_path)
                    resized_files.append(output_path)
                    print(f"Resized: {filename} -> {target_size}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nCompleted! Resized {len(resized_files)} images. Saved to: {output_folder}")
    return resized_files


def convert_colors_to_grayscale(input_folder=None, output_folder=None, target_size=(224,224)):
    """
    Convert all wafer map images in input_folder to grayscale masks:
    - Green (0,255,0) -> 1 (gray)
    - Red   (255,0,0) -> 2 (white)
    - Others          -> 1 (gray)
    """
    if input_folder is None:
        input_folder = config['data']['wafer_map_png']  # ใช้ default จาก config
    if output_folder is None:
        output_folder = config['data']['wafer_map_grayscale']

    os.makedirs(output_folder, exist_ok=True)
    supported_extensions = ('.png', '.jpg', '.jpeg')
    grayscale_files = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    arr = np.array(img)

                    # สร้าง mask
                    mask = np.zeros(arr.shape[:2], dtype=np.uint8)

                    # หา pixel ที่เป็นเขียวและแดง
                    green_mask = (arr[:,:,0] == 0) & (arr[:,:,1] == 0) & (arr[:,:,2] == 255)
                    red_mask   = (arr[:,:,0] == 255) & (arr[:,:,1] == 0) & (arr[:,:,2] == 0)

                    mask[green_mask] = 1   # เทา
                    mask[red_mask]   = 2   # ขาว

                    # สเกลค่า (0=ดำ, 127=เทา, 254=ขาว)
                    mask_img = Image.fromarray(mask * 127)

                    # Resize (ถ้ามี target_size)
                    if target_size:
                        mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)

                    # Save
                    output_path = os.path.join(output_folder, filename)
                    mask_img.save(output_path)
                    grayscale_files.append(output_path)
                    print(f"Converted: {filename} -> grayscale")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nCompleted! Converted {len(grayscale_files)} images. Saved to: {output_folder}")
    return grayscale_files
