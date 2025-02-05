how to run
On Linux:
python main.py \
 --train_path "./UIT-VSFC_train.csv" \
 --dev_path "./UIT-VSFC_dev.csv" \
 --test_path "./UIT-VSFC_test.csv" \
 --stopwords_path "stopwords-vi_news.txt"

python main.py \
 --train_path "./UIT-VSFC_train.csv" \
 --dev_path "./UIT-VSFC_dev.csv" \
 --test_path "./UIT-VSFC_test.csv" \
 --stopwords_path "stopwords-vi_news.txt"\
 --use_dash
On Windows:

python main.py --train_path "./UIT-VSFC_train.csv" --dev_path "./UIT-VSFC_dev.csv" --test_path "./UIT-VSFC_test.csv" --stopwords_path "stopwords-vi_news.txt"

install anaconda on Ubuntu:
chmod 777 install_anaconda.sh
sudo ./setup_anaconda.sh

python main.py \
 --train_path "./UIT-VSFC_train.csv" \
 --dev_path "./UIT-VSFC_dev.csv" \
 --test_path "./UIT-VSFC_test.csv" \
 --stopwords_path "stopwords-vi_news.txt" \
 --use_simple
