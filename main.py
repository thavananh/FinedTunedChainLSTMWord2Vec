# Ví dụ sử dụng:
from multiprocessing import process
from DataLoader import DataLoader
from Package import PackageInstaller
from Preprocessing import VietnameseTextPreprocessor

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
    # train_text_tokens = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=True) for text in train_text]
    train_text_preprocessed = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=False) for text in train_text]
    # test_text_tokens = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=True) for text in test_text]
    test_text_preprocessed = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=False) for text in test_text]
    # dev_text_tokens = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=True) for text in dev_text]
    dev_text_preprocessed = [preprocessor.preprocess_text_vietnamese_to_tokens(text, isReturnTokens=False) for text in dev_text]
    
    # print('train_text_tokens:', train_text_tokens)
    print('train_text_preprocessed:', train_text_preprocessed)
    # print('test_text_tokens:', test_text_tokens)
    # print('test_text_preprocessed:', test_text_preprocessed)
    # print('dev_text_tokens:', dev_text_tokens)
    # print('dev_text_preprocessed:', dev_text_preprocessed)