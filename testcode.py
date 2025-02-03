import argparse
from multiprocessing import process
import os
import pickle
import csv
import subprocess
import sys

from sympy import use

from CustomerHyperModel import CustomHyperModel
from DataLoader import DataLoader

from Preprocessing import VietnameseTextPreprocessor
from Word2Vec import Word2VecModel
from main import save_to_csv
parser = argparse.ArgumentParser(description='Train model for Vietnamese text classification.')
parser.add_argument('--train_path', type=str, required=True, help='Training data path')
parser.add_argument('--dev_path', type=str, required=True, help='Development data path')
parser.add_argument('--test_path', type=str, required=True, help='Testing data path')
parser.add_argument('--stopwords_path', type=str, required=True, help='Stopwords data path')
parser.add_argument('--use_dash', action='store_true', help='Use dash in preprocessor')
parser.add_argument('--use_simple', action='store_true', help='Simple spliting text')
args = parser.parse_args()

print("User's Train path:", args.train_path)
print("User's Dev path:", args.dev_path)
print("User's Test path:", args.test_path)
print("User's Stopwords path:", args.stopwords_path)
print("User's Use dash:", args.use_dash)
print("User's Use simple:", args.use_simple)

# Cài đặt các gói cần thiết

# print(f'Packages: {os.system("pip list")}')

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
unpreprocessed_label_train = loader['unprocessed_label_train']
unpreprocessed_label_test = loader['unprocessed_label_test']
unpreprocessed_label_dev = loader['unprocessed_label_dev']

# Tiền xử lý văn bản
preprocessor = VietnameseTextPreprocessor(stopwords_path=args.stopwords_path)
print('Warning: if is simple is set to True, the function do not use any type of tokenize but just using split method and it will return normal form.')

# Xử lý các tập dữ liệu
process_text = lambda texts: [
    preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=True, isUsingDash=args.use_dash, isSimple=args.use_simple) 
    for text in texts
]

train_text_tokens = process_text(train_text)
test_text_tokens = process_text(test_text)
dev_text_tokens = process_text(dev_text)

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
save_to_csv(train_text_preprocessed, unpreprocessed_label_train, 'processed_train.csv')
save_to_csv(test_text_preprocessed, unpreprocessed_label_test, 'processed_test.csv')
save_to_csv(dev_text_preprocessed, unpreprocessed_label_dev, 'processed_dev.csv')
print('Đã lưu các tập dữ liệu đã xử lý vào CSV.')