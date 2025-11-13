import os
import tarfile
import shutil
from pathlib import Path

# ------------------------------
# CONFIG PATH
# ------------------------------
INPUT_GZ_DIR = Path(r"C:\Users\User\Documents\GitHub\testwork\little_wafer\Data\GZ_Files")
ZIP_EXTRACT_DIR = Path(r"C:\Users\User\Documents\GitHub\testwork\little_wafer\Data\Zipfile-STDF")
OUTPUT_PRR_DIR = Path(r"C:\Users\User\Documents\GitHub\testwork\little_wafer\Data\Output-PRR")


# ------------------------------
# 1️⃣ ฟังก์ชันสำหรับแตกไฟล์ .tar.gz ทั้งหมด
# ------------------------------
def extract_all_tar_gz(input_dir=INPUT_GZ_DIR, output_dir=ZIP_EXTRACT_DIR):
    """แตกไฟล์ .tar.gz ทั้งหมดจาก input_dir ไปไว้ใน output_dir"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gz_files = list(input_dir.glob("*.tar.gz"))
    if not gz_files:
        print(f"⚠️ ไม่พบไฟล์ .tar.gz ในโฟลเดอร์: {input_dir}")
        return

    print(f"🔍 พบทั้งหมด {len(gz_files)} ไฟล์ .tar.gz ใน {input_dir}")

    for gz_file in gz_files:
        try:
            filename = gz_file.name
            lot_folder = filename.replace(".std.tar.gz", "")
            lot_output_folder = output_dir / lot_folder
            lot_output_folder.mkdir(exist_ok=True)

            with tarfile.open(gz_file, "r:gz") as tar:
                tar.extractall(path=lot_output_folder)

            print(f"✅ แตกไฟล์เรียบร้อย: {filename} → {lot_output_folder}")

        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {gz_file.name}: {e}")

    print("\n🎯 เสร็จสิ้นการแตกไฟล์ทั้งหมด\n")


# ------------------------------
# 2️⃣ ฟังก์ชันสำหรับย้ายไฟล์ .std ไปไว้ในโฟลเดอร์ lotname ที่ Output-PRR
# ------------------------------
def move_std_files_to_lot_folders(input_dir=ZIP_EXTRACT_DIR, output_dir=OUTPUT_PRR_DIR):
    """
    ค้นหาไฟล์ .std ทั้งหมดใน input_dir (รวม subfolders)
    แล้วคัดลอกไปยัง output_dir/<lotname>/
    lotname จะดึงจากชื่อไฟล์ เช่น ZA539411_001_S11P... → ZA539411
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    std_files = list(input_dir.rglob("*.std"))
    if not std_files:
        print(f"⚠️ ไม่พบไฟล์ .std ใน: {input_dir}")
        return

    print(f"🔍 พบทั้งหมด {len(std_files)} ไฟล์ .std")

    for std_file in std_files:
        try:
            # ตัวอย่างชื่อไฟล์: ZA539411_001_S11P_20251003021933_M5116B16P03_WFTB33-01.std
            filename = std_file.name
            lotname = filename.split("_")[0]  # ดึง 'ZA539411'
            lot_output_folder = output_dir / lotname
            lot_output_folder.mkdir(exist_ok=True)

            # คัดลอกไฟล์ไปยังโฟลเดอร์ของ lot นั้นๆ
            dest_file = lot_output_folder / filename
            shutil.copy2(std_file, dest_file)

            print(f"✅ คัดลอกไฟล์: {filename} → {lot_output_folder}")

        except Exception as e:
            print(f"❌ Error ย้ายไฟล์ {std_file}: {e}")

    print("\n🎯 เสร็จสิ้นการคัดลอกไฟล์ทั้งหมด\n")


# ------------------------------
# 3️⃣ ตัวอย่างการเรียกใช้ฟังก์ชัน
# ------------------------------
if __name__ == "__main__":
    # ขั้นตอนที่ 1: แตกไฟล์ทั้งหมด
    extract_all_tar_gz()

    # ขั้นตอนที่ 2: ย้าย .std ไปยังโฟลเดอร์ Output-PRR
    move_std_files_to_lot_folders()
