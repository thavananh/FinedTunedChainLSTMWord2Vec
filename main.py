import argparse
from multiprocessing import process
import pickle
import csv

from DataLoader import DataLoader
from Package import PackageInstaller
from Preprocessing import VietnameseTextPreprocessor
from Word2Vec import Word2VecModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Model import CustomModel

def save_to_csv(text_list, label_list, file_name):
    with open(file_name, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'label'])
        for text, label in zip(text_list, label_list):
            writer.writerow([text, label])

def main():
    # Thiết lập các tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Train model for Vietnamese text classification.')
    parser.add_argument('--train_path', type=str, required=True, help='Training data path')
    parser.add_argument('--dev_path', type=str, required=True, help='Development data path')
    parser.add_argument('--test_path', type=str, required=True, help='Testing data path')
    parser.add_argument('--stopwords_path', type=str, required=True, help='Stopwords data path')
    parser.add_argument('--use_dash', action='store_true', help='Use dash in preprocessor')
    
    args = parser.parse_args()

    print("User's Train path:", args.train_path)
    print("User's Dev path:", args.dev_path)
    print("User's Test path:", args.test_path)
    print("User's Stopwords path:", args.stopwords_path)
    print("User's Use dash:", args.use_dash)

    # Cài đặt các gói cần thiết
    installer = PackageInstaller(['pyvi', 'underthesea'])
    installer.install_packages()
    print("Đã cài các gói:", installer.list_packages())

    # Nạp dữ liệu
    data_loader = DataLoader(args.train_path, args.dev_path, args.test_path)
    data_loader.load_data()
    loader = data_loader.get_processed_data()
    
    # Trích xuất dữ liệu
    train_text = loader['train_text']
    train_label = loader['train_labels']
    test_text = loader['test_text']
    test_label = loader['test_labels']
    dev_text = loader['dev_text']
    dev_label = loader['dev_labels']

    # Tiền xử lý văn bản
    preprocessor = VietnameseTextPreprocessor(stopwords_path=args.stopwords_path)
    use_dash = args.use_dash
    
    # Xử lý các tập dữ liệu
    process_text = lambda texts, flag: [
        preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=True, isUsingDash=use_dash) 
        for text in texts
    ]
    
    train_text_tokens = process_text(train_text, use_dash)
    test_text_tokens = process_text(test_text, use_dash)
    dev_text_tokens = process_text(dev_text, use_dash)

    train_text_preprocessed = [' '.join(tokens) for tokens in train_text_tokens]
    test_text_preprocessed = [' '.join(tokens) for tokens in test_text_tokens]
    dev_text_preprocessed = [' '.join(tokens) for tokens in dev_text_tokens]

    print('top 5 train_text_tokens:', train_text_tokens[:5])
    print('top 5 train_text_preprocessed:', train_text_preprocessed[:5])
    print('top 5 test_text_tokens:', test_text_tokens[:5])
    print('top 5 test_text_preprocessed:', test_text_preprocessed[:5])
    print('top 5 dev_text_tokens:', dev_text_tokens[:5])
    print('top 5 dev_text_preprocessed:', dev_text_preprocessed[:5])

    # Lưu dữ liệu đã xử lý
    save_to_csv(train_text_preprocessed, train_label, 'processed_train.csv')
    save_to_csv(test_text_preprocessed, test_label, 'processed_test.csv')
    save_to_csv(dev_text_preprocessed, dev_label, 'processed_dev.csv')
    print('Đã lưu các tập dữ liệu đã xử lý vào CSV.')

    # Huấn luyện mô hình Word2Vec
    print("\nTraining CBOW model...")
    model_cbow = Word2VecModel(sg=0)
    model_cbow.train(train_text_tokens, epochs=15)
    model_cbow.save('model_cbow')

    print("\nTraining Skip-Gram model...")
    model_sg = Word2VecModel(sg=1)
    model_sg.train(train_text_tokens, epochs=15)
    model_sg.save('model_sg')

    # Chuẩn bị dữ liệu cho model
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(train_text_tokens)
    
    # Padding sequences
    max_len = 110
    train_features = pad_sequences(tokenizer.texts_to_sequences(train_text_preprocessed), maxlen=max_len)
    test_features = pad_sequences(tokenizer.texts_to_sequences(test_text_preprocessed), maxlen=max_len)
    dev_features = pad_sequences(tokenizer.texts_to_sequences(dev_text_preprocessed), maxlen=max_len)
    
    # Lưu tokenizer
    pickle.dump(tokenizer, open("tokenizer_data.pkl", "wb"))

    # Khởi tạo và huấn luyện mô hình

    print(tokenizer.word_index)

    print("\nPreparing embedding matrix...")
    model_sg = Word2VecModel()
    model_sg.load_model('model_sg.word2vec')
    embedding_matrix = model_sg.get_embedding_matrix(tokenizer)

    print("\nBuilding model...")
    model = CustomModel(len(tokenizer.word_index)+1, embedding_matrix)
    model.build_model()
    model.compile_model()
    print(train_label.shape)
    print(dev_label.shape)
    model.train(train_features, train_label, dev_features, dev_label)
    
    # Đánh giá mô hình
    model.evaluate_model(test_features, test_label)
    preds = model.predict(test_features)
    # preds = tf.round(preds).numpy()
    model.generate_classification_report(test_label, preds)
    model.plot_confusion_matrix(test_label, preds)

if __name__ == "__main__":
    main()