from PIL import Image
import os

def resize_img(input_folder, output_folder=None, target_size=(224,224)):
    """ Resize all images in the input folder to the target size. """
    if output_folder is None:
        output_folder = input_folder + '_resized'
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