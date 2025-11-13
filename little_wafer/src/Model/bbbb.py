# src/plot_predictions.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_prediction_bar(
    csv_path="Output/predictions/test_predictions.csv", 
    save_path="Output/predictions/prediction_bar.png",
    save_summary=True
):
    """
     พล็อตกราฟแท่งแสดงจำนวนภาพที่โมเดลทำนายแต่ละคลาส (fail pattern)
    - แกน X: ชื่อคลาส (pred_label)
    - แกน Y: จำนวนภาพที่อยู่ใน test set
    """

    # โหลดไฟล์ผลการทำนาย
    df = pd.read_csv(csv_path)
    if "pred_label" not in df.columns:
        raise ValueError("❌ ไม่พบคอลัมน์ 'pred_label' ในไฟล์ CSV")

    # สรุปจำนวนภาพต่อคลาส
    class_counts = df["pred_label"].value_counts().sort_index()
    total_images = len(df)
    print(f"📊 พบภาพทั้งหมด {total_images} ภาพใน test set")

    # วาดกราฟแท่ง
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=class_counts.index, 
        y=class_counts.values, 
        palette="tab10"
    )

    # ✅ แสดงตัวเลขบนแท่ง
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

    plt.title("Distribution of Predicted Defect Patterns", fontsize=16)
    plt.xlabel("Defect Pattern (Predicted Class)", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # ✅ บันทึกกราฟ
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Bar plot saved to: {save_path}")
    plt.close()

    # ✅ บันทึก summary CSV (optional)
    if save_summary:
        summary_path = os.path.splitext(save_path)[0] + "_summary.csv"
        summary_df = class_counts.reset_index()
        summary_df.columns = ["pred_label", "count"]
        summary_df["percentage"] = (summary_df["count"] / total_images * 100).round(2)
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary CSV saved to: {summary_path}")

if __name__ == "__main__":
    plot_prediction_bar("Output/predictions/test_predictions.csv")
