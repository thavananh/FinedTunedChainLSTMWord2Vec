import os
import time  # Thêm module time để sử dụng sleep
import google.generativeai as genai
import pandas as pd
import ollama
import subprocess

#Download model
import ollama

# Cấu hình API key cho Google Generative AI
process = subprocess.Popen("ollama serve", shell=True) #runs on a different thread
genai.configure(api_key="Your key")
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Hàm tạo sinh paraphrase từ câu gốc
def generate_paraphrases(sentence):
    response = model.generate_content(
        f"Paraphrase câu này '{sentence}' theo 5 cách khác nhau. Bạn chỉ cần trả về câu, không giải thích, không sử dụng số thứ tự, không sử dụng các ký tự phân dòng, chỉ cần xuống dòng."
    )
    paraphrases = response.text.strip().split("\n")
    return paraphrases

def generate_paraphrases_ollama(sentence):
    response = ollama.chat(model='hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Q8_0',messages = [
        {"role": "user", "content": f"Paraphrase câu này '{sentence}' theo 5 cách khác nhau. Bạn chỉ cần trả về câu, không giải thích, không sử dụng số thứ tự, không sử dụng các ký tự phân dòng."},
    ])
    paraphrases = response['message']['content'].strip().split("\n")
    return paraphrases

# Đọc dữ liệu từ các file CSV
train_data = pd.read_csv('UIT-VSFC_train.csv').dropna()
test_data = pd.read_csv('UIT-VSFC_test.csv').dropna()
dev_data = pd.read_csv('UIT-VSFC_dev.csv').dropna()

# Tạo thư mục để lưu trạng thái nếu chưa tồn tại
os.makedirs("checkpoints", exist_ok=True)

# Hàm tạo sinh dữ liệu mới với tính năng lưu trạng thái và delay
def augment_data(data, checkpoint_file):
    # Kiểm tra xem có file checkpoint không
    if os.path.exists(checkpoint_file):
        checkpoint = pd.read_csv(checkpoint_file)
        processed_indices = set(checkpoint['index'])
    else:
        checkpoint = pd.DataFrame(columns=['index', 'sents', 'sentiments'])
        processed_indices = set()

    augmented_data = []
    for idx, row in data.iterrows():
        if idx in processed_indices:
            # Nếu câu đã được xử lý, bỏ qua
            continue

        sentence = row['sents']
        sentiment = row['sentiments']

        # Tạo sinh 5 câu paraphrase
        try:
            paraphrases = generate_paraphrases(sentence)
        except Exception as e:
            print(f"Error generating paraphrases for sentence: {sentence}. Error: {e}")
            paraphrases = []

        # Thêm câu gốc vào danh sách
        augmented_data.append({'sents': sentence, 'sentiments': sentiment})

        # Thêm các câu paraphrase vào danh sách với cùng nhãn sentiment
        for paraphrase in paraphrases:
            augmented_data.append({'sents': paraphrase, 'sentiments': sentiment})

        # Lưu trạng thái sau mỗi câu
        new_row = pd.DataFrame([{'index': idx, 'sents': sentence, 'sentiments': sentiment}])
        checkpoint = pd.concat([checkpoint, new_row], ignore_index=True)
        checkpoint.to_csv(checkpoint_file, index=False)

        # Thêm delay 1.5 giây giữa các request
        time.sleep(1.5)

    return pd.DataFrame(augmented_data)

# Tạo sinh dữ liệu mới cho từng tập
augmented_train_data = augment_data(train_data, "checkpoints/train_checkpoint.csv")
augmented_test_data = augment_data(test_data, "checkpoints/test_checkpoint.csv")
augmented_dev_data = augment_data(dev_data, "checkpoints/dev_checkpoint.csv")

# Lưu dữ liệu mới vào file CSV
augmented_train_data.to_csv('UIT-VSFC_train_augmented.csv', index=False)
augmented_test_data.to_csv('UIT-VSFC_test_augmented.csv', index=False)
augmented_dev_data.to_csv('UIT-VSFC_dev_augmented.csv', index=False)

print("Dữ liệu đã được tạo sinh và lưu thành công!")