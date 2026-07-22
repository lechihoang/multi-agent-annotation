import pandas as pd
import random

# Load data
orig = pd.read_csv('data/ViCTSD_train.csv').fillna('')
new = pd.read_csv('data/ViCTSD_train_reannotated.csv').fillna('')

# Lọc bỏ những dòng không được gán nhãn
valid_indices = new['Constructiveness'] != -1
orig = orig[valid_indices].reset_index(drop=True)
new = new[valid_indices].reset_index(drop=True)

orig_labels = orig['Constructiveness'].astype(int)
new_labels = new['Constructiveness'].astype(int)

# Thống kê
total = len(orig)
h0_a0 = ((orig_labels == 0) & (new_labels == 0)).sum()
h1_a1 = ((orig_labels == 1) & (new_labels == 1)).sum()
h1_a0 = ((orig_labels == 1) & (new_labels == 0)).sum()
h0_a1 = ((orig_labels == 0) & (new_labels == 1)).sum()

print("=== BẢNG THỐNG KÊ SO SÁNH NHÃN TRÊN TẬP TRAIN ===")
print(f"Tổng số câu: {total}")
print(f"Giống nhau hoàn toàn: {h0_a0 + h1_a1} câu ({((h0_a0 + h1_a1)/total)*100:.2f}%)")
print(f"  - Người: 0 | AI: 0 => {h0_a0} câu")
print(f"  - Người: 1 | AI: 1 => {h1_a1} câu")
print(f"Khác nhau: {h1_a0 + h0_a1} câu ({((h1_a0 + h0_a1)/total)*100:.2f}%)")
print(f"  - Người: 1 (Constructive) nhưng AI giáng xuống 0 (Non-constructive) => {h1_a0} câu")
print(f"  - Người: 0 (Non-constructive) nhưng AI nâng lên 1 (Constructive) => {h0_a1} câu")

print("\n\n=== VÍ DỤ: Người gán 1 (Constructive) nhưng AI gán 0 (Non-constructive) ===")
print("Đây là những câu mà con người đánh giá cao, nhưng AI thấy 'chưa đủ trình độ' để làm Constructive:")
h1_a0_texts = orig[(orig_labels == 1) & (new_labels == 0)]['Comment'].tolist()
random.seed(42)
for i, text in enumerate(random.sample(h1_a0_texts, min(5, len(h1_a0_texts)))):
    print(f"{i+1}. {text}")

print("\n\n=== VÍ DỤ: Người gán 0 (Non-constructive) nhưng AI gán 1 (Constructive) ===")
print("Đây là những câu mà con người bỏ qua, nhưng AI lại 'moi' ra được thông tin / lập luận hữu ích:")
h0_a1_texts = orig[(orig_labels == 0) & (new_labels == 1)]['Comment'].tolist()
for i, text in enumerate(random.sample(h0_a1_texts, min(5, len(h0_a1_texts)))):
    print(f"{i+1}. {text}")

