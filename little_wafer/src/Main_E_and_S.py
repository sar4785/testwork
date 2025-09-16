from convert_stdf_to_prr.ExtractZip import ExtractZip
from convert_stdf_to_prr.STD_to_PRR import PRRConverter

if __name__ == "__main__":
    # ✅ ขั้นตอนที่ 1: แตก zip → ได้ไฟล์ .std (เลือก session filter ได้)
    SESSION = "S11P"  # เปลี่ยนเป็น None ถ้าอยากดึงทั้งหมด
    std_files = ExtractZip.run(session_filter=SESSION)

    # ✅ ขั้นตอนที่ 2: แปลง .std → PRR.csv
    if std_files:
        PRRConverter.convert_stdf_to_prr(
            input_root=ExtractZip.output_folder, # อ่านจากโฟลเดอร์ที่ ExtractZip บันทึก
            output_root=PRRConverter.OUTPUT_ROOT,
            debug=True
        )
    else:
        print("❌ ไม่มีไฟล์ .std ให้ประมวลผล")    
