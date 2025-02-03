import re
import unicodedata
from underthesea import word_tokenize, sent_tokenize

def preprocess_fn(text):
    '''
    Preprocessing text
    '''
    text = unicodedata.normalize('NFKC', str(text))
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = " ".join([word_tokenize(sent, format='text') for sent in  sent_tokenize(text)])
    return text

print(preprocess_fn("môn học cần phân chia hợp lý giữa các phần , phần nào khó cần nhiều thời gian để học nó , phần nào đã học thì chỉ ra những điểm đặc biệt cũng như thời gian học phần đó rút ngắn lại để các phần chưa học khác sẽ có thời gian nhiều hơn ."))