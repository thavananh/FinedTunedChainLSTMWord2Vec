# Ví dụ sử dụng:
from multiprocessing import process
import pickle

from sympy import im
from DataLoader import DataLoader
from Package import PackageInstaller
from Preprocessing import VietnameseTextPreprocessor
import csv
from Word2Vec import Word2VecModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Model import CustomModel

if __name__ == "__main__":
    # Khởi tạo đối tượng với danh sách gói cần cài
    installer = PackageInstaller(['pyvi', 'underthesea'])

    # Cài đặt các gói
    installer.install_packages()

    # In ra danh sách gói đã cài
    print(installer.list_packages())


    train_path = './UIT-VSFC_train.csv'
    dev_path = './UIT-VSFC_dev.csv'
    test_path = './UIT-VSFC_test.csv'
    data_loader = DataLoader(train_path, dev_path, test_path)
    data_loader.load_data()
    loader = data_loader.get_processed_data()
    train_text = loader['train_text']
    train_label = loader['train_labels']
    test_text = loader['test_text']
    test_label = loader['test_labels']
    dev_text = loader['dev_text']
    dev_label = loader['dev_labels']
    print('len(train_text):', len(train_text))
    print('len(train_label):', len(train_label))
    print('len(test_text):', len(test_text))
    print('len(test_label):', len(test_label))
    print('len(dev_text):', len(dev_text))
    print('len(dev_label):', len(dev_label))

    preprocessor = VietnameseTextPreprocessor()
    
    train_text_tokens = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=True, isUsingDash=False) for text in train_text]
    train_text_preprocessed = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=False, isUsingDash=False) for text in train_text]
    test_text_tokens = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=True, isUsingDash=False) for text in test_text]
    test_text_preprocessed = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=False, isUsingDash=False) for text in test_text]
    dev_text_tokens = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=True, isUsingDash=False) for text in dev_text]
    dev_text_preprocessed = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=False, isUsingDash=False) for text in dev_text]
    
    print('train_text_tokens:', train_text_tokens)
    print('train_text_preprocessed:', train_text_preprocessed)
    print('len(train_text_preprocessed):', len(train_text_preprocessed))
    print('test_text_tokens:', test_text_tokens)
    print('test_text_preprocessed:', test_text_preprocessed)
    print('dev_text_tokens:', dev_text_tokens)
    print('dev_text_preprocessed:', dev_text_preprocessed)

    # Lưu các list vào file csv
    def save_to_csv(text_list, label_list, file_name):
        with open(file_name, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['text', 'label'])
            for text, label in zip(text_list, label_list):
                writer.writerow([text, label])

    save_to_csv(train_text_preprocessed, train_label, 'processed_train.csv')
    save_to_csv(test_text_preprocessed, test_label, 'processed_test.csv')
    save_to_csv(dev_text_preprocessed, dev_label, 'processed_dev.csv')

    print('Train, test, and dev sets have been saved to CSV files.')

    # Khởi tạo model CBOW
    model_cbow = Word2VecModel(sg=0)
    model_cbow.train(train_text_tokens, epochs=15)
    model_cbow.save('model_cbow')

    # Khởi tạo model Skip-Gram
    model_sg = Word2VecModel(sg=1)
    model_sg.train(train_text_tokens, epochs=15)
    model_sg.save('model_sg')

    length = []
    for x in train_text_preprocessed:
        length.append(len(x.split()))
    print('max length of sentence in train', max(length))
    
    length = []
    for x in test_text_preprocessed:
        length.append(len(x.split()))
    print('max length of sentence in test', max(length))

    length = []
    for x in dev_text_preprocessed:
        length.append(len(x.split()))
    print('max length of sentence in dev', max(length))
    
    tokenizer_data = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')

    tokenizer_data.fit_on_texts(train_text_preprocessed)

    tokenized_data_text_train = tokenizer_data.texts_to_sequences(train_text_preprocessed)
    train_features = pad_sequences(tokenized_data_text_train, maxlen=110)

    tokenized_data_text_test = tokenizer_data.texts_to_sequences(test_text_preprocessed)
    test_features = pad_sequences(tokenized_data_text_test, maxlen=110)

    tokenized_data_text_dev = tokenizer_data.texts_to_sequences(dev_text_preprocessed)
    dev_features = pad_sequences(tokenized_data_text_dev, maxlen=110)

    pickle.dump(tokenizer_data, open("tokenizer_data.pkl", "wb"))
    data_vocab_size = len(tokenizer_data.word_index) + 1

    print("input data shape:", train_features.shape)
    print("data_vocab_size:", data_vocab_size)
    print("training sample:", len(train_features))
    print("validation sample:", len(dev_features))
    print("test sample:", len(test_features))

    print(tokenizer_data.word_index)

    model_sg = Word2VecModel()
    model_sg.load_model('model_sg.word2vec')
    embedding_matrix = model_sg.get_embedding_matrix()
    print('word2vec dict', model_sg.get_vocab_dict())

    # Example usage:
    model = CustomModel(data_vocab_size, embedding_matrix)
    model.build_model()
    model.compile_model()
    model.train(train_features, train_label, dev_features, dev_label)
    model.evaluate_model(test_features, test_label)
    preds = model.predict(test_features)
    model.generate_classification_report(test_features, preds)
    model.plot_confusion_matrix(test_features, preds)