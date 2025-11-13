# src/utils/data_utils.py

import random, shutil
from pathlib import Path

def merge_random_and_normal(
    train_or_val_dir,
    random_name="random",
    normal_name="normal",
    target_count=1000,
    remove_original_normal=False,
    seed=42,
):
    """
    รวมภาพจาก random + normal เข้าด้วยกันเป็น class เดียวชื่อ 'random'
    โดยเลือกภาพจากแต่ละคลาสครึ่งต่อครึ่ง (เช่น 500/500)
    """
    import random, shutil
    from pathlib import Path

    random.seed(seed)
    train_dir = Path(train_or_val_dir).resolve()
    src_random = train_dir / random_name
    src_normal = train_dir / normal_name
    dst_random = train_dir / random_name

    # ✅ ตรวจสอบว่ามีอย่างน้อย 1 โฟลเดอร์ที่มีภาพ
    imgs_random = list(src_random.glob("*.png")) + list(src_random.glob("*.jpg")) if src_random.exists() else []
    imgs_normal = list(src_normal.glob("*.png")) + list(src_normal.glob("*.jpg")) if src_normal.exists() else []

    if len(imgs_random) == 0 and len(imgs_normal) == 0:
        print(f"⚠️ ไม่มีภาพในทั้ง random และ normal ที่ {train_or_val_dir} — ข้ามการรวม")
        return

    # ✅ ถ้า random ไม่มีภาพเลยแต่ normal มี → ใช้ภาพจาก normal ทั้งหมดแทน
    if len(imgs_random) == 0 and len(imgs_normal) > 0:
        print(f"⚠️ ไม่มี random → ใช้ normal ทั้งหมด {len(imgs_normal)} ภาพแทน")
        dst_random.mkdir(parents=True, exist_ok=True)
        for img in imgs_normal:
            shutil.copy(img, dst_random / img.name)
        return

    # ✅ ถ้า normal ไม่มีภาพ → ใช้ random เดิมทั้งหมด
    if len(imgs_normal) == 0 and len(imgs_random) > 0:
        print(f"⚠️ ไม่มี normal → ใช้ random เดิมทั้งหมด {len(imgs_random)} ภาพ")
        return

    # ✅ ปลอดภัยแล้ว ค่อยรวม
    dst_random.mkdir(parents=True, exist_ok=True)

    need_each = target_count // 2
    sel_random = random.sample(imgs_random, min(len(imgs_random), need_each))
    sel_normal = random.sample(imgs_normal, min(len(imgs_normal), need_each))

    combined = sel_random + sel_normal
    random.shuffle(combined)

    if len(combined) < target_count:
        pool = [p for p in (imgs_random + imgs_normal) if p not in combined]
        random.shuffle(pool)
        combined += pool[: target_count - len(combined)]

    if len(combined) > target_count:
        combined = random.sample(combined, target_count)

    # ⚠️ ลบเฉพาะไฟล์ที่ต้องแทนที่ (ถ้า folder random มีอยู่แล้ว)
    for old in dst_random.glob("*"):
        old.unlink(missing_ok=True)

    for p in combined:
        if not p.exists():
            print(f"⚠️ Skip missing file: {p}")
            continue
        dst = dst_random / p.name
        if dst.exists():
            base, ext = p.stem, p.suffix
            i = 1
            while (dst_random / f"{base}_{i}{ext}").exists():
                i += 1
            dst = dst_random / f"{base}_{i}{ext}"
        shutil.copy(p, dst)

    if remove_original_normal and src_normal.exists():
        shutil.rmtree(src_normal, ignore_errors=True)

    print(f"✅ รวมภาพสำเร็จ → {dst_random} (ทั้งหมด {len(list(dst_random.glob('*')))} รูป)")

    
def refill_train_data(train_random_dir, target_count=1000):
    """
    ถ้าภาพใน train/random น้อยกว่าที่กำหนด → generate เพิ่มจากภาพเดิม (copy ซ้ำ)
    """
    train_random_dir = Path(train_random_dir)
    imgs = list(train_random_dir.glob("*"))
    if len(imgs) >= target_count:
        return        
    
    print(f"🧩 Refill train set: {len(imgs)} → {target_count}")
    while len(list(train_random_dir.glob("*"))) < target_count:
        for p in imgs:
            if len(list(train_random_dir.glob("*"))) >= target_count:
                break
            dst = train_random_dir / f"{p.stem}_dup{random.randint(0,9999)}{p.suffix}"
            shutil.copy(p, dst)
    print(f"✅ Fill data in train{target_count} pictures.")
 
 
        