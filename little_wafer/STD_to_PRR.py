# STD_to_PRR.py
import os
from pathlib import Path
import pandas as pd
from io import StringIO
import pystdf.V4 as v4
from pystdf.IO import Parser
from pystdf.Writers import TextWriter

# 📂 ตำแหน่งโฟลเดอร์ input/output
INPUT_ROOT = r"C:\Users\User\Desktop\Wafer-classification-DCNN\convert_stdf_to_csv\Output-STDF" 
OUTPUT_ROOT = r"C:\Users\User\Desktop\Wafer-classification-DCNN\convert_stdf_to_csv\Output-PRR"

class PRRConverter:
    INPUT_ROOT = INPUT_ROOT #folder ที่เก็บ .std
    OUTPUT_ROOT = OUTPUT_ROOT #folder ที่เก็บ .csv
    
    @staticmethod
    def convert_stdf_to_prr(input_root, output_root, debug=False):
        """
        Convert .std files to PRR.csv
        :param input_root: Path ที่เก็บ .std
        :param output_root: Path ที่ต้องการบันทึก PRR.csv
        :param debug: แสดง log เพิ่มเติม
        """
        os.makedirs(output_root, exist_ok=True)

        filelist = list(Path(input_root).rglob("*.std"))
        combined_data = []

        if not filelist:
            print(f"⚠️ ไม่พบไฟล์ .std ใน: {input_root}")
            return pd.DataFrame()

        for file in filelist:
            filename = file.name
            lotname = filename.split("_")[0] 
            lot_output_dir = Path(output_root) / lotname
            lot_output_dir.mkdir(exist_ok=True)
            
            try:
                # parse std → atdf
                with file.open("rb") as f:
                    p = Parser(inp=f)
                    captured_std_out = StringIO()
                    p.addSink(TextWriter(captured_std_out))
                    p.parse()
                    atdf = captured_std_out.getvalue()

                atdf_lines = atdf.split("\n")

                # --- Extract PRR only ---
                prr_data = []
                for record_type in v4.records:
                    record_name = record_type.name.split(".")[-1].upper()
                    if record_name != "PRR":
                        continue
                    curr = [line for line in atdf_lines if line.startswith(record_name)]
                    for line in curr:
                        parts = line.split("|")
                        # สร้าง dict จาก fieldMap ของ PRR
                        prr_dict = dict(zip([f[0] for f in record_type.fieldMap], parts[1:]))
                        prr_data.append({
                            "X_COORD": int(prr_dict.get("X_COORD", -9999)),
                            "Y_COORD": int(prr_dict.get("Y_COORD", -9999)),
                            "SOFT_BIN": int(prr_dict.get("SOFT_BIN", -1)),
                            "HARD_BIN": int(prr_dict.get("HARD_BIN", -1)),
                            "SITE_NUM": int(prr_dict.get("SITE_NUM", -1)),
                            "PART_ID": prr_dict.get("PART_ID", ""),
                            "SourceFile": filename,
                        })

                # บันทึก PRR.csv สำหรับแต่ละ lot
                if prr_data:
                    prr_df = pd.DataFrame(prr_data)
                    #สร้างชื่อไฟล์จากชื่อ .std เดิม แต่เปลี่ยนเป็น .csv
                    csv_name = filename.replace(".std", ".csv")
                    prr_csv_path = lot_output_dir / csv_name
                    prr_df.to_csv(prr_csv_path, index=False)
                    print(f"✅ Saved PRR: {prr_csv_path}")
                    combined_data.append(prr_df)
                else:
                    print(f"⚠️ ไม่พบ PRR ในไฟล์: {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
