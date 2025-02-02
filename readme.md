how to run
On Linux:
python main.py \
 --train_path "./UIT-VSFC_train.csv" \
 --dev_path "./UIT-VSFC_dev.csv" \
 --test_path "./UIT-VSFC_test.csv" \
 --stopwords_path "stopwords-vi_news.txt"
On Windows:

python main.py --train_path "./UIT-VSFC_train.csv" --dev_path "./UIT-VSFC_dev.csv" --test_path "./UIT-VSFC_test.csv" --stopwords_path "stopwords-vi_news.txt"

install anaconda on Ubuntu:
curl -O https://gist.githubusercontent.com/NCKH-collab/9e71c0555c73e02572907b7ef8f5ddb0/raw/31c0eb1c57557a2ac4d0ed574ad5eea73e1e62cd/install_anaconda.sh
chmod +x install_anaconda.sh
./install_anaconda.sh