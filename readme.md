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
 --train_aug_path "./train_aug_merge_uit_vsfc.csv" \
 --dev_aug_path "./dev_aug_merge_uit_vsfc.csv" \
 --test_aug_path "./test_aug_merge_uit_vsfc.csv" \
 --stopwords_path "stopwords-vi_news.txt" \
 --use_aug_data \
 --telegram_group_id "-4788105046" \
 --telegram_bot_id "7773727606" \

if you want to run ollama please follow this script "curl -fsSL https://ollama.com/install.sh | sh" after that run "pip install ollama"
if you want a fast, memory efficent model, please use the model from Unsloth. It's very good, work like magic.
run "ollama pull hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Q8_0"
