import os
from PRR_to_PNG_and_Resize.PRR_to_PNG import PRRToPNGConverter
from PRR_to_PNG_and_Resize.Resize_PNG import resize_img

USER_PATH = r"C:\Users\Phimprasert\Desktop\Wafer-classification-DCNN"
INPUT_PRR_DIR = os.path.join(USER_PATH, "convert_stdf_to_prr", "Output-PRR")
OUTPUT_PNG_DIR = os.path.join(USER_PATH, "PRR_to_PNG_and_Resize", "Wafer-Map")
OUTPUT_RESIZED_DIR = OUTPUT_PNG_DIR + "_resized"

def main():
    print("🚀 Starting Main Pipeline: PRR to PNG + Resize")
    # Step 1: PRR → PNG
    print("\n📈 Step 1: Converting PRR CSVs to PNG wafer maps...")
    PRRToPNGConverter.main(input_dir=INPUT_PRR_DIR, output_dir=OUTPUT_PNG_DIR)

    png_files = [f for f in os.listdir(OUTPUT_PNG_DIR) if f.lower().endswith(".png")]
    if not png_files:
        print("❌ No PNGs generated! Check PRR input directory.")
        return
    print(f"✅ Found {len(png_files)} PNG files in {OUTPUT_PNG_DIR}")

    # Step 2: Resize
    print("\n🖼️ Step 2: Resizing PNGs to 224x224...")
    resized_files = resize_img(
        input_folder=OUTPUT_PNG_DIR,
        output_folder=OUTPUT_RESIZED_DIR,
        target_size=(224, 224)
    )
    print(f"\n🎉 Pipeline completed! Resized {len(resized_files)} images to {OUTPUT_RESIZED_DIR}")

if __name__ == "__main__":
    main()