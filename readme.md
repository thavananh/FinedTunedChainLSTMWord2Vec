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
curl -O https://gist.githubusercontent.com/NCKH-collab/9e71c0555c73e02572907b7ef8f5ddb0/raw/622a0cb4e8a7318e215b064bab2f9f7bce15fe98/install_anaconda.sh
chmod +x install_anaconda.sh
./install_anaconda.sh


python main.py \
     --train_path "./UIT-VSFC_train.csv" \
     --dev_path "./UIT-VSFC_dev.csv" \
     --test_path "./UIT-VSFC_test.csv" \
     --stopwords_path "stopwords-vi_news.txt" \
    --use_simple