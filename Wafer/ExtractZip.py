import os
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path

# 📂 โฟลเดอร์ต้นทางและปลายทาง
ZIP_FOLDER = r"C:\Users\User\Desktop\Wafer-classification-DCNN\convert_stdf_to_csv\Zipfile-STDF"
OUTPUT_FOLDER = r"C:\Users\User\Desktop\Wafer-classification-DCNN\convert_stdf_to_csv\Output-STDF"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class ExtractZip:
    zip_folder = ZIP_FOLDER #folder ที่แตกไฟล์ zip
    output_folder = OUTPUT_FOLDER #folder ที่เก็บ .std ที่แตกออกมา
    
    @staticmethod
    def extract_nested(file_path, temp_dir, session_filter=None):
        """
        แตกไฟล์ซ้อน ๆ จนเจอ .std
        """
        os.makedirs(temp_dir, exist_ok=True)

        # ✅ ถ้าเจอไฟล์ .std
        if file_path.lower().endswith(".std"):
            file_path_str = str(Path(file_path))
            if session_filter and session_filter.upper() not in file_path_str.upper():
                print(f"⏭️ ข้ามไฟล์ (ไม่ตรง {session_filter}): {file_path}")
                return None

            filename = Path(file_path).name
            lotname = filename.split('_')[0]
            lot_output = os.path.join(OUTPUT_FOLDER, lotname)
            os.makedirs(lot_output, exist_ok=True)

            dest_path = os.path.join(lot_output, filename)
            if not os.path.exists(dest_path):
                shutil.copy(file_path, dest_path)
                print(f"✅ คัดลอก .std ({session_filter or 'ทั้งหมด'}): {dest_path}")
            return dest_path

        # 📦 zip
        elif file_path.lower().endswith(".zip"):
            print(f"📦 แตก ZIP: {Path(file_path).name}")
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    ExtractZip.extract_nested(
                        os.path.join(root, f),
                        os.path.join(temp_dir, "nested_zip"),
                        session_filter
                    )

        # 📦 tar.gz
        elif file_path.lower().endswith((".tar.gz", ".tgz")):
            print(f"📦 แตก TAR.GZ: {Path(file_path).name}")
            try:
                with tarfile.open(file_path, "r:gz") as tar_ref:
                    tar_ref.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for f in files:
                        ExtractZip.extract_nested(
                            os.path.join(root, f),
                            os.path.join(temp_dir, "nested_tar"),
                            session_filter
                        )
            except Exception as e:
                print(f"❌ ไม่สามารถแตก TAR.GZ: {file_path} -> {e}")

        # 📂 gz (เดี่ยว)
        elif file_path.lower().endswith(".gz"):
            print(f"📦 แตก GZ: {Path(file_path).name}")
            try:
                out_path = Path(temp_dir) / Path(file_path).stem
                with gzip.open(file_path, "rb") as f_in, open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return ExtractZip.extract_nested(
                    str(out_path),
                    os.path.join(temp_dir, "nested_gz"),
                    session_filter
                )
            except Exception as e:
                print(f"❌ ไม่สามารถแตก GZ: {file_path} -> {e}")

        return None

    @staticmethod
    def run(session_filter=None):
        """ค้นหาและแตกไฟล์ .std ทั้งหมด"""
        temp_root = Path(OUTPUT_FOLDER) / "_temp"
        if temp_root.exists():
            shutil.rmtree(temp_root)

        std_files = []
        for item in os.listdir(ZIP_FOLDER):
            item_path = Path(ZIP_FOLDER) / item
            if item_path.is_file():
                print(f"\n🔎 ประมวลผล: {item}")
                extracted_std = ExtractZip.extract_nested(
                    str(item_path),
                    str(temp_root / item_path.stem),
                    session_filter
                )
                if extracted_std:
                    std_files.append(extracted_std)

        if not std_files:
            print(f"❌ ไม่พบ .std (filter={session_filter})")
        else:
            print(f"\n🎉 เสร็จสิ้น! พบ {len(std_files)} ไฟล์ .std")
            print(f"📂 อยู่ที่: {OUTPUT_FOLDER}")

        if temp_root.exists():
            shutil.rmtree(temp_root)

        return std_files
