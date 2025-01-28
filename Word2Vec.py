import multiprocessing
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from torch import embedding

class Word2VecModel:
    """
    Class đóng gói mô hình Word2Vec với các chức năng:
    - Khởi tạo mô hình
    - Huấn luyện mô hình
    - Fine-tune mô hình
    - Lưu mô hình ở 2 định dạng
    """
    
    def __init__(self, sg=0, vector_size=300, window=7, min_count=10, workers=-1, sample=1e-3, alpha=0.025, min_alpha=0.0007, negative=5):
        """
        Khởi tạo các tham số cho mô hình
        """
        self.sg = sg                          # 0 = CBOW, 1 = Skip-Gram
        self.vector_size = vector_size        # Kích thước vector
        self.window = window                  # Kích thước cửa sổ ngữ cảnh
        self.min_count = min_count            # Tần suất xuất hiện tối thiểu
        self.sample = sample                  # Tỷ lệ giảm mẫu cho các từ tần suất cao
        self.alpha = alpha                    # Learning rate ban đầu
        self.min_alpha = min_alpha            # Learning rate tối thiểu
        self.negative = negative              # Số lượng từ nhiễu (negative sampling)
        self.workers = multiprocessing.cpu_count() if workers == -1 else workers
        self.model = None
    
    def build_model(self):
        """Khởi tạo architecture mô hình"""
        self.model = Word2Vec(
            sg=self.sg,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sample=self.sample,
            alpha=self.alpha,
            min_alpha=self.min_alpha,
            negative=self.negative,
            workers=self.workers
        )
    
    def train(self, tokens, epochs=30):
        """
        Huấn luyện mô hình với dữ liệu đầu vào
        
        Args:
            tokens (list): Danh sách các câu đã tokenize
            epochs (int): Số lần huấn luyện
        """
        if self.model is None:
            self.build_model()
            
        self.model.build_vocab(tokens)
        self.model.train(tokens, total_examples=len(tokens), epochs=epochs)
    
    def fine_tune(self, new_tokens, epochs=10, alpha=None):
        """
        Fine-tune mô hình với dữ liệu mới
        
        Args:
            new_tokens (list): Danh sách các câu mới đã tokenize
            epochs (int): Số lần huấn luyện bổ sung
            alpha (float): Learning rate mới cho quá trình fine-tuning (nếu có)
        """
        if self.model is None:
            raise Exception("Model chưa được huấn luyện")
        
        if alpha:
            self.model.alpha = alpha
            self.model.min_alpha = alpha
        
        self.model.build_vocab(new_tokens, update=True)
        self.model.train(new_tokens, total_examples=len(new_tokens), epochs=epochs)
    
    def save(self, filename):
        """
        Lưu mô hình ở 2 định dạng:
        - .word2vec: Định dạng của Gensim
        - .bin: Định dạng nhị phân Word2Vec
        """
        if self.model:
            self.model.save(f"{filename}.word2vec")
            self.model.wv.save_word2vec_format(f"{filename}.bin", binary=True)
        else:
            raise Exception("Model chưa được huấn luyện")
    def get_embedding_matrix(self):
        """
        Trả về ma trận embedding của mô hình Word2Vec
        """
        embeddings_index = {}
        for w in self.model.wv.key_to_index.keys():
            embeddings_index[w] = self.model.wv[w]

        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def load_model(self, path, isBinary=False):
        if not isBinary:
            self.model = Word2Vec.load(path)
        else:
            self.model = KeyedVectors.load(path, mmap='r')

    def get_vocab_dict(self):
        return self.model.wv.key_to_index
    
    def get_vocab_size(self):
        return len(self.model.wv.key_to_index)
