import argparse
from multiprocessing import process
import os
import pickle
import csv
import subprocess
import sys

from sklearn.metrics import classification_report

from CustomHyperModel import CustomHyperModel
from DataLoader import DataLoader

from Preprocessing import VietnameseTextPreprocessor
from Word2Vec import Word2VecModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.Model_0 import CustomModel_0
import keras_tuner as kt

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report
import numpy as np


class ClassificationReportCallback(Callback):
    def __init__(
        self, X_val, y_val, X_test, y_test, label_names
    ):  # Thêm X_test, y_test
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.label_names = label_names

    def on_epoch_end(self, epoch, logs=None):
        # Validation Report
        y_pred_val = self.model.predict(self.X_val)
        y_pred_val_classes = np.argmax(y_pred_val, axis=1)
        y_true_val = np.argmax(self.y_val, axis=1)

        report_val = classification_report(
            y_true_val,
            y_pred_val_classes,
            target_names=self.label_names,
            zero_division=0,
        )
        print(f"\nEpoch {epoch + 1} - Classification Report (Validation):\n")
        print(report_val)

        # Test Report
        y_pred_test = self.model.predict(self.X_test)
        y_pred_test_classes = np.argmax(y_pred_test, axis=1)
        y_true_test = np.argmax(self.y_test, axis=1)

        report_test = classification_report(
            y_true_test,
            y_pred_test_classes,
            target_names=self.label_names,
            zero_division=0,
        )
        print(f"\nEpoch {epoch + 1} - Classification Report (Test):\n")
        print(report_test)


def save_to_csv(text_list, label_list, file_name):
    with open(file_name, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        for text, label in zip(text_list, label_list):
            writer.writerow([text, label])


def install_requirements(file_path="requirements.txt"):
    try:
        # Kiểm tra xem file requirements.txt có tồn tại không
        with open(file_path, "r") as f:
            print(f"Đang đọc file: {file_path}")

        # Chạy lệnh pip install -r requirements.txt
        print("Đang cài đặt các thư viện từ requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", file_path])
        print("Cài đặt thành công!")

    except FileNotFoundError:
        print(f"File {file_path} không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
    except subprocess.CalledProcessError as e:
        print(f"Có lỗi xảy ra khi cài đặt các thư viện: {e}")
    except Exception as e:
        print(f"Lỗi không mong đợi: {e}")


def main():
    # Thiết lập các tham số dòng lệnh
    parser = argparse.ArgumentParser(
        description="Train model for Vietnamese text classification."
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Training data path"
    )
    parser.add_argument(
        "--dev_path", type=str, required=True, help="Development data path"
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Testing data path"
    )
    parser.add_argument(
        "--stopwords_path", type=str, required=True, help="Stopwords data path"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name"
    )
    parser.add_argument(
        "--epoch_tune", type=int, required=True, help="Tune's Epoch"
    )
    parser.add_argument(
        "--telegram_bot_id", type=str, required=True, help="Telegram bot id"
    )
    parser.add_argument(
        "--telegram_group_id", type=str, required=True, help="Telegram group id"
    )
    parser.add_argument(
        "--use_dash", action="store_true", help="Use dash in preprocessor"
    )
    parser.add_argument(
        "--use_simple", action="store_true", help="Simple spliting text"
    )
     


    args = parser.parse_args()

    print("User's Train path:", args.train_path)
    print("User's Dev path:", args.dev_path)
    print("User's Test path:", args.test_path)
    print("User's Stopwords path:", args.stopwords_path)
    print("User's model name: ", args.model_name)
    print("User's epoch tune: ", args.epoch_tune)
    print("User's Telegram bot id: ", args.telegram_bot_id)
    print("User's Telegram group id: ", args.telegram_group_id)
    print("User's Use dash:", args.use_dash)
    print("User's Use simple:", args.use_simple)

    # Cài đặt các gói cần thiết

    print(f'Packages: {os.system("pip list")}')

    # Nạp dữ liệu
    data_loader = DataLoader(args.train_path, args.dev_path, args.test_path)
    data_loader.load_data()
    loader = data_loader.get_processed_data()

    # Trích xuất dữ liệu
    train_text = loader["train_text"]
    train_label = loader["train_labels"]
    test_text = loader["test_text"]
    test_label = loader["test_labels"]
    dev_text = loader["dev_text"]
    dev_label = loader["dev_labels"]
    unpreprocessed_label_train = loader["unprocessed_label_train"]
    unpreprocessed_label_test = loader["unprocessed_label_test"]
    unpreprocessed_label_dev = loader["unprocessed_label_dev"]

    # Tiền xử lý văn bản
    preprocessor = VietnameseTextPreprocessor(stopwords_path=args.stopwords_path)
    use_dash = args.use_dash

    # Xử lý các tập dữ liệu
    process_text = lambda texts: [
        preprocessor.preprocess_text_vietnamese_to_tokens(
            text, isReturnTokens=True, isUsingDash=use_dash, isSimple=args.use_simple
        )
        for text in texts
    ]

    train_text_tokens = process_text(train_text)
    test_text_tokens = process_text(test_text)
    dev_text_tokens = process_text(dev_text)

    train_text_preprocessed = [" ".join(tokens) for tokens in train_text_tokens]
    test_text_preprocessed = [" ".join(tokens) for tokens in test_text_tokens]
    dev_text_preprocessed = [" ".join(tokens) for tokens in dev_text_tokens]

    print("top 5 train_text_tokens:", train_text_tokens[:5])
    print("top 5 train_text_preprocessed:", train_text_preprocessed[:5])
    print("top 5 test_text_tokens:", test_text_tokens[:5])
    print("top 5 test_text_preprocessed:", test_text_preprocessed[:5])
    print("top 5 dev_text_tokens:", dev_text_tokens[:5])
    print("top 5 dev_text_preprocessed:", dev_text_preprocessed[:5])

    # Lưu dữ liệu đã xử lý
    save_to_csv(
        train_text_preprocessed, unpreprocessed_label_train, "processed_train.csv"
    )
    save_to_csv(test_text_preprocessed, unpreprocessed_label_test, "processed_test.csv")
    save_to_csv(dev_text_preprocessed, unpreprocessed_label_dev, "processed_dev.csv")
    print("Đã lưu các tập dữ liệu đã xử lý vào CSV.")

    # Huấn luyện mô hình Word2Vec
    print("\nTraining CBOW model...")

    if not args.use_dash:
        dev_train_combined = train_text_preprocessed + dev_text_preprocessed
        train_text_tokens_from_sent = [sent.split() for sent in dev_train_combined]
    train_text_tokens_from_sent = train_text_tokens + dev_text_tokens

    print("top 5 train_text_tokens_from_sent:", train_text_tokens_from_sent[:5])

    model_cbow = Word2VecModel(sg=0)
    model_cbow.train(train_text_tokens_from_sent, epochs=15)
    model_cbow.save("model_cbow")

    print("\nTraining Skip-Gram model...")
    model_sg = Word2VecModel(sg=1)
    model_sg.train(train_text_tokens_from_sent, epochs=15)
    model_sg.save("model_sg")

    print(model_sg.get_vocab_dict())

    # Chuẩn bị dữ liệu cho model
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(train_text_tokens_from_sent)

    # Padding sequences
    max_len = 130
    train_features = pad_sequences(
        tokenizer.texts_to_sequences(train_text_preprocessed), maxlen=max_len
    )
    test_features = pad_sequences(
        tokenizer.texts_to_sequences(test_text_preprocessed), maxlen=max_len
    )
    dev_features = pad_sequences(
        tokenizer.texts_to_sequences(dev_text_preprocessed), maxlen=max_len
    )

    # Lưu tokenizer
    pickle.dump(tokenizer, open("tokenizer_data.pkl", "wb"))

    # Khởi tạo và huấn luyện mô hình

    print(tokenizer.word_index)

    print("\nPreparing embedding matrix...")
    model_sg = Word2VecModel()
    model_sg.load_model("model_sg.word2vec")
    embedding_matrix = model_sg.get_embedding_matrix(tokenizer)

    # print("\nBuilding model...")
    # model = CustomModel_0(
    #     len(tokenizer.word_index) + 1, embedding_matrix, input_length=max_len
    # )
    # model.build_model()
    # model.compile_model()
    # print(train_label.shape)
    # print(dev_label.shape)
    # model.train(train_features, train_label, dev_features, dev_label, epochs=2)

    # # Đánh giá mô hình
    # model.evaluate_model(test_features, test_label)
    # preds = model.predict(test_features)
    # preds = tf.round(preds).numpy()
    # model.generate_classification_report(test_label, preds)
    # model.plot_confusion_matrix(test_label, preds, is_print_terminal=True)

    hypermodel = CustomHyperModel(
        w2v_corpus=train_text_tokens_from_sent,
        tokenizer_data=tokenizer,
        input_length=130,  # Match your input sequence length
        X_train=train_features,
        y_train=train_label,
        X_val=dev_features,
        y_val=dev_label,
        X_test=test_features,
        y_test=test_label,
        model_name=args.model_name,
        epoch_num=args.epoch_tune,
        telegram_bot_id=args.telegram_bot_id,
        group_chat_id=args.telegram_group_id,
    )
    tuner = kt.Hyperband(
        hypermodel=hypermodel,
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="hyper_tuning",
        project_name="sentiment_analysis",
        hyperband_iterations=1,
        overwrite=True,
    )
    tuner.search()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(
        f"""
    Best Word2Vec parameters:
    - Architecture: {'Skip-gram' if best_hps.get('w2v_sg') else 'CBOW'}
    - Vector Size: {best_hps.get('w2v_vector_size')}
    - Window Size: {best_hps.get('w2v_window')}
    - Min Count: {best_hps.get('w2v_min_count')}
    """
    )

    # Retrieve best model
    best_model = tuner.get_best_models(num_models=1)[0]


if __name__ == "__main__":
    main()
