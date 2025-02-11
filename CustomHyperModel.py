import asyncio
from datetime import datetime
import multiprocessing
import os
import keras_tuner as kt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import Word2Vec
import tensorflow as tf
from models.KimCNN import KimCNNModel
from models.CZhouLSTMCNN import CZhouLSTMCNNModel
from models.Model_0 import CustomModel_0  # Assuming Model_0.py exists
from models.Model_1 import CustomModel_1
from models.Model_2 import CustomModel_2
from models.Model_3 import CustomModel_3
from models.PZhouLSTMCNN import PZhouLSTMCNNModel
from utils.Attribute import *  # Assuming Attribute.py exists
import yaml
import telegram 


async def send_report_via_telegram(report_filename, telegram_bot_id, group_chat_id):
    # Khởi tạo bot với token của bạn
    bot = telegram.Bot(token=telegram_bot_id)
    telegram_chat_id = group_chat_id
    
    # Mở file ở chế độ đọc nhị phân và gửi file
    with open(report_filename, 'rb') as doc:
        await bot.send_document(chat_id=telegram_chat_id, document=doc)
    print("Báo cáo đã được gửi qua Telegram.")

class CustomHyperModel(kt.HyperModel):
    def __init__(
        self,
        w2v_corpus,
        tokenizer_data,
        input_length,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        model_name,
        epoch_num,
        telegram_bot_id,
        group_chat_id
    ):
        super().__init__()
        self.w2v_corpus = w2v_corpus
        self.tokenizer_data = tokenizer_data
        self.input_length = input_length
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.config = self.load_config("./config/model.yaml")  # Load main config
        if self.config is None:
            raise ValueError(f"Failed to load config file.")
        self.model_config = self.config.get(
            self.model_name, {}
        )  # Load model-specific config
        if self.model_config is None:
            raise ValueError(f"Failed to load config for model: {model_name}")
        self.w2v_config = self.config.get("Word2Vec", {})  # Load model-specific config
        if self.w2v_config is None:
            raise ValueError(f"Failed to load config for model: Word2Vec")
        self.epoch_num = epoch_num
        self.telegram_bot_id = telegram_bot_id
        self.group_chat_id = group_chat_id

    def load_config(self, config_path):
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_path}")
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def _get_hp_value(self, hp, param_config):
        """Helper function to get hyperparameter values based on config."""
        param_type = param_config.get("type")
        param_name = param_config.get("name")

        if param_type == "int":
            return hp.Int(
                param_name,
                min_value=param_config.get("range", [0])[0],  # Handle ranges
                max_value=param_config.get("range", [0, 1])[1],
                step=param_config.get("step", 1),
            )
        elif param_type == "float":
            return hp.Float(
                param_name,
                min_value=param_config.get("range", [0.0])[0],  # Handle ranges
                max_value=param_config.get("range", [0.0, 1.0])[1],
                step=param_config.get("step"),
                sampling=param_config.get("sampling"),
            )
        elif param_type == "choice":
            return hp.Choice(param_name, values=param_config.get("options", []))
        else:
            raise ValueError(f"Unknown hyperparameter type: {param_type}")

    def build(self, hp):
        # Word2Vec Hyperparameters
        w2v_params = {}
        for param_name, param_config in self.w2v_config.items():
            if (
                isinstance(param_config, dict) and "type" in param_config
            ):  # Check if it's a hyperparameter definition
                w2v_params[param_config.get("name")] = self._get_hp_value(
                    hp, param_config
                )
            else:
                w2v_params[param_name] = (
                    param_config  # Use the value directly if not a hyperparameter definition
                )
        # Train Word2Vec
        w2v_model = Word2Vec(**w2v_params, workers=multiprocessing.cpu_count())
        w2v_model.build_vocab(self.w2v_corpus)
        w2v_model.train(
            self.w2v_corpus,
            total_examples=len(self.w2v_corpus),
            epochs=w2v_params.get("epochs", 30),  # Use get with default
        )

        # Generate Embedding Matrix
        data_vocab_size = len(self.tokenizer_data.word_index) + 1
        embedding_matrix = np.random.normal(
            0, 0.05, (data_vocab_size, w2v_params["vector_size"])
        )

        for word, i in self.tokenizer_data.word_index.items():
            if i >= data_vocab_size:
                continue
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]

        custom_model = None

        if self.model_name == "CustomModel_0":
            hp_custom = {}
            for key, value in self.model_config.items():
                hp_custom[value.get("name")] = self._get_hp_value(hp, value)
            custom_model = CustomModel_0(
                data_vocab_size=data_vocab_size,
                embedding_matrix=embedding_matrix,
                input_length=self.input_length,
                dropout_combine=hp_custom["dropout_combine"],
                dropout_features=hp_custom["dropout_features"],
                cnn_attributes_1=CnnAtribute(
                    filter_size=hp_custom["cnn_1_filter_size"],
                    kernel_size=hp_custom["cnn_1_kernel_size"],
                    padding=hp_custom["cnn_1_padding"],
                    activation=hp_custom["cnn_1_activation"],
                    # dropout_rate=hp_custom["cnn_1_dropout_rate"],
                ),
                cnn_attributes_2=CnnAtribute(
                    filter_size=hp_custom["cnn_2_filter_size"],
                    kernel_size=hp_custom["cnn_2_kernel_size"],
                    padding=hp_custom["cnn_2_padding"],
                    activation=hp_custom["cnn_2_activation"],
                    # dropout_rate=hp_custom["cnn_2_dropout_rate"],
                ),
                cnn_attributes_3=CnnAtribute(
                    filter_size=hp_custom["cnn_3_filter_size"],
                    kernel_size=hp_custom["cnn_3_kernel_size"],
                    padding=hp_custom["cnn_3_padding"],
                    activation=hp_custom["cnn_3_activation"],
                    # dropout_rate=hp_custom["cnn_3_dropout_rate"],
                ),
                cnn_attributes_4=CnnAtribute(
                    filter_size=hp_custom["cnn_4_filter_size"],
                    kernel_size=hp_custom["cnn_4_kernel_size"],
                    padding=hp_custom["cnn_4_padding"],
                    activation=hp_custom["cnn_4_activation"],
                    # dropout_rate=hp_custom["cnn_4_dropout_rate"],
                ),
                lstm_attributes_1=LSTMAttribute(
                    units=hp_custom["lstm_1_units"],
                    dropout_rate=hp_custom["lstm_1_dropout_rate"],
                ),
                lstm_attributes_2=LSTMAttribute(
                    units=hp_custom["lstm_2_units"],
                    dropout_rate=hp_custom["lstm_2_dropout_rate"],
                ),
                multi_head_attention_attributes=MultiHeadAttentionAttribute(
                    num_heads=hp_custom["multi_head_attention_num_heads"],
                    key_dim=hp_custom["multi_head_attention_key_dim"],
                    dropout_rate=hp_custom["multi_head_attention_dropout_rate"],
                ),
                dense_attributes_1=DenseAttribute(
                    units=hp_custom["dense_1_units"],
                    # dropout_rate=hp_custom["dense_1_dropout_rate"],
                    activation=hp_custom["dense_1_activation"],
                ),
                dense_attributes_2=DenseAttribute(
                    units=hp_custom["dense_2_units"],
                    # dropout_rate=hp_custom["dense_2_dropout_rate"],
                    activation=hp_custom["dense_2_activation"],
                ),
                dense_attributes_3=DenseAttribute(
                    units=3,
                    # dropout_rate=hp_custom["dense_3_dropout_rate"],
                    activation=hp_custom["dense_3_activation"],
                ),
            )
        elif self.model_name == "CustomModel_1":
            hp_custom = {}
            print(hp_custom)
            for key, value in self.model_config.items():
                hp_custom[value.get("name")] = self._get_hp_value(hp, value)
            custom_model = CustomModel_1(
                data_vocab_size=data_vocab_size,
                embedding_matrix=embedding_matrix,
                input_length=self.input_length,
                dropout_features=hp_custom["dropout_features"],
                dropout_combine=hp_custom["dropout_combine"],
                cnn_attributes_1=CnnAtribute(
                    filter_size=hp_custom["cnn_1_filter_size"],
                    kernel_size=hp_custom["cnn_1_kernel_size"],
                    padding=hp_custom["cnn_1_padding"],
                    activation=hp_custom["cnn_1_activation"],
                    # dropout_rate=hp_custom["cnn_1_dropout_rate"],
                ),
                cnn_attributes_2=CnnAtribute(
                    filter_size=hp_custom["cnn_2_filter_size"],
                    kernel_size=hp_custom["cnn_2_kernel_size"],
                    padding=hp_custom["cnn_2_padding"],
                    activation=hp_custom["cnn_2_activation"],
                    # dropout_rate=hp_custom["cnn_2_dropout_rate"],
                ),
                lstm_attributes_1=LSTMAttribute(
                    units=hp_custom["lstm_1_units"],
                    dropout_rate=hp_custom["lstm_1_dropout_rate"],
                ),
                lstm_attributes_2=LSTMAttribute(
                    units=hp_custom["lstm_2_units"],
                    dropout_rate=hp_custom["lstm_2_dropout_rate"],
                ),
                multi_head_attention_attributes=MultiHeadAttentionAttribute(
                    num_heads=hp_custom["multi_head_attention_num_heads"],
                    key_dim=hp_custom["multi_head_attention_key_dim"],
                    dropout_rate=hp_custom["multi_head_attention_dropout_rate"],
                ),
                dense_attributes_1=DenseAttribute(
                    units=hp_custom["dense_1_units"],
                    # dropout_rate=hp_custom["dense_1_dropout_rate"],
                    activation=hp_custom["dense_1_activation"],
                ),
                dense_attributes_3=DenseAttribute(
                    units=3,
                    # dropout_rate=hp_custom["dense_3_dropout_rate"],
                    activation=hp_custom["dense_3_activation"],
                ),
            )
        elif self.model_name == "CustomModel_2":
            hp_custom = {}
            for key, value in self.model_config.items():
                hp_custom[value.get("name")] = self._get_hp_value(hp, value)
            custom_model = CustomModel_2(
                data_vocab_size=data_vocab_size,
                embedding_matrix=embedding_matrix,
                input_length=self.input_length,
                dropout_combine=hp_custom["dropout_combine"],
                dropout_features=hp_custom["dropout_features"],
                lstm_attributes_1=LSTMAttribute(
                    units=hp_custom["lstm_1_units"],
                    dropout_rate=hp_custom["lstm_1_dropout_rate"],
                ),
                lstm_attributes_2=LSTMAttribute(
                    units=hp_custom["lstm_2_units"],
                    dropout_rate=hp_custom["lstm_2_dropout_rate"],
                ),
                lstm_attributes_3=LSTMAttribute(
                    units=hp_custom["lstm_3_units"],
                    dropout_rate=hp_custom["lstm_3_dropout_rate"],
                ),
                multi_head_attention_attributes=MultiHeadAttentionAttribute(
                    num_heads=hp_custom["multi_head_attention_num_heads"],
                    key_dim=hp_custom["multi_head_attention_key_dim"],
                    dropout_rate=hp_custom["multi_head_attention_dropout_rate"],
                ),
                dense_attributes_1=DenseAttribute(
                    units=hp_custom["dense_1_units"],
                    # dropout_rate=hp_custom["dense_1_dropout_rate"],
                    activation=hp_custom["dense_1_activation"],
                ),
                dense_attributes_2=DenseAttribute(
                    units=hp_custom["dense_2_units"],
                    # dropout_rate=hp_custom["dense_2_dropout_rate"],
                    activation=hp_custom["dense_2_activation"],
                ),
                dense_attributes_3=DenseAttribute(
                    units=3,
                    # dropout_rate=hp_custom["dense_3_dropout_rate"],
                    activation=hp_custom["dense_3_activation"],
                ),
            )
        elif self.model_name == "CustomModel_3":
            hp_custom = {}
            for key, value in self.model_config.items():
                hp_custom[value.get("name")] = self._get_hp_value(hp, value)
            custom_model = CustomModel_3(
                data_vocab_size=data_vocab_size,
                embedding_matrix=embedding_matrix,
                input_length=self.input_length,
                dropout_combine=hp_custom["dropout_combine"],
                dropout_features=hp_custom["dropout_features"],
                dropout_attention_pooled=hp_custom["dropout_attention_pooled"],
                attention_weight_activation=hp_custom["attention_weight_activation"],
                cnn_attributes_1=CnnAtribute(
                    filter_size=hp_custom["cnn_1_filter_size"],
                    kernel_size=hp_custom["cnn_1_kernel_size"],
                    padding=hp_custom["cnn_1_padding"],
                    activation=hp_custom["cnn_1_activation"],
                    # dropout_rate=hp_custom["cnn_1_dropout_rate"],
                ),
                cnn_attributes_2=CnnAtribute(
                    filter_size=hp_custom["cnn_2_filter_size"],
                    kernel_size=hp_custom["cnn_2_kernel_size"],
                    padding=hp_custom["cnn_2_padding"],
                    activation=hp_custom["cnn_2_activation"],
                    # dropout_rate=hp_custom["cnn_2_dropout_rate"],
                ),
                lstm_attributes_1=LSTMAttribute(
                    units=hp_custom["lstm_1_units"],
                    dropout_rate=hp_custom["lstm_1_dropout_rate"],
                ),
                multi_head_attention_attributes=MultiHeadAttentionAttribute(
                    num_heads=hp_custom["multi_head_attention_num_heads"],
                    key_dim=hp_custom["multi_head_attention_key_dim"],
                    dropout_rate=hp_custom["multi_head_attention_dropout_rate"],
                ),
                dense_attributes_1=DenseAttribute(
                    units=hp_custom["dense_1_units"],
                    # dropout_rate=hp_custom["dense_1_dropout_rate"],
                    activation=hp_custom["dense_1_activation"],
                ),
                dense_attributes_3=DenseAttribute(
                    units=3,
                    # dropout_rate=hp_custom["dense_3_dropout_rate"],
                    activation=hp_custom["dense_3_activation"],
                ),
            )
        elif self.model_name == "CZhouLSTMCNN":
            hp_custom = {}
            for key, value in self.model_config.items():
                hp_custom[value.get("name")] = self._get_hp_value(hp, value)
            custom_model = CZhouLSTMCNNModel(
                data_vocab_size=data_vocab_size,
                embedding_matrix=embedding_matrix,
                input_length=self.input_length,
                dropout_combine=hp_custom["dropout_combine"],
                dropout_features=hp_custom["dropout_features"],
                cnn_attributes_1=CnnAtribute(
                    filter_size=hp_custom["cnn_1_filter_size"],
                    kernel_size=hp_custom["cnn_1_kernel_size"],
                    padding=hp_custom["cnn_1_padding"],
                    activation=hp_custom["cnn_1_activation"],
                    # dropout_rate=hp_custom["cnn_1_dropout_rate"],
                ),
                cnn_attributes_2=CnnAtribute(
                    filter_size=hp_custom["cnn_2_filter_size"],
                    kernel_size=hp_custom["cnn_2_kernel_size"],
                    padding=hp_custom["cnn_2_padding"],
                    activation=hp_custom["cnn_2_activation"],
                    # dropout_rate=hp_custom["cnn_2_dropout_rate"],
                ),
                cnn_attributes_3=CnnAtribute(
                    filter_size=hp_custom["cnn_3_filter_size"],
                    kernel_size=hp_custom["cnn_3_kernel_size"],
                    padding=hp_custom["cnn_3_padding"],
                    activation=hp_custom["cnn_3_activation"],
                    # dropout_rate=hp_custom["cnn_3_dropout_rate"],
                ),
                cnn_attributes_4=CnnAtribute(
                    filter_size=hp_custom["cnn_4_filter_size"],
                    kernel_size=hp_custom["cnn_4_kernel_size"],
                    padding=hp_custom["cnn_4_padding"],
                    activation=hp_custom["cnn_4_activation"],
                    # dropout_rate=hp_custom["cnn_4_dropout_rate"],
                ),
                cnn_2d_attribute_1=Cnn2DAttribute(
                    filter_size=hp_custom["cnn_2d_1_filter_size"],
                    kernel_size=(
                        hp_custom["cnn_2d_1_kernel_size_height"],
                        hp_custom["cnn_2d_1_kernel_size_width"],
                    ),
                    padding=hp_custom["cnn_2d_1_padding"],
                    activation=hp_custom["cnn_2d_1_activation"],
                    # dropout_rate=hp_custom["cnn_2d_1_dropout_rate"],
                ),
                cnn_2d_attribute_2=Cnn2DAttribute(
                    filter_size=hp_custom["cnn_2d_2_filter_size"],
                    kernel_size=(
                        hp_custom["cnn_2d_2_kernel_size_height"],
                        hp_custom["cnn_2d_1_kernel_size_width"],
                    ),
                    padding=hp_custom["cnn_2d_2_padding"],
                    activation=hp_custom["cnn_2d_2_activation"],
                    # dropout_rate=hp_custom["cnn_2d_2_dropout_rate"],
                ),
                lstm_attributes_1=LSTMAttribute(
                    units=hp_custom["lstm_1_units"],
                    dropout_rate=hp_custom["lstm_1_dropout_rate"],
                ),
                lstm_attributes_2=LSTMAttribute(
                    units=hp_custom["lstm_2_units"],
                    dropout_rate=hp_custom["lstm_2_dropout_rate"],
                ),
                dense_attributes_1=DenseAttribute(
                    units=hp_custom["dense_1_units"],
                    # dropout_rate=hp_custom["dense_1_dropout_rate"],
                    activation=hp_custom["dense_1_activation"],
                ),
                dense_attributes_3=DenseAttribute(
                    units=3,
                    # dropout_rate=hp_custom["dense_3_dropout_rate"],
                    activation=hp_custom["dense_3_activation"],
                ),
            )
        elif self.model_name == "KimCNN":
            hp_custom = {}
            for key, value in self.model_config.items():
                hp_custom[value.get("name")] = self._get_hp_value(hp, value)
            custom_model = KimCNNModel(
                data_vocab_size=data_vocab_size,
                embedding_matrix=embedding_matrix,
                input_length=self.input_length,
                # dropout_combine=hp_custom["dropout_combine"],
                dropout_features=hp_custom["dropout_features"],
                cnn_2d_attribute_1=Cnn2DAttribute(
                    filter_size=hp_custom["cnn_2d_1_filter_size"],
                    kernel_size=(
                        hp_custom["cnn_2d_1_kernel_size_height"],
                        hp_custom["cnn_2d_1_kernel_size_width"],
                    ),
                    padding=hp_custom["cnn_2d_1_padding"],
                    activation=hp_custom["cnn_2d_1_activation"],
                    # dropout_rate=hp_custom["cnn_2d_1_dropout_rate"],
                ),
                cnn_2d_attribute_2=Cnn2DAttribute(
                    filter_size=hp_custom["cnn_2d_2_filter_size"],
                    kernel_size=(
                        hp_custom["cnn_2d_2_kernel_size_height"],
                        hp_custom["cnn_2d_2_kernel_size_width"],
                    ),
                    padding=hp_custom["cnn_2d_2_padding"],
                    activation=hp_custom["cnn_2d_2_activation"],
                    # dropout_rate=hp_custom["cnn_2d_2_dropout_rate"],
                ),
                dense_attributes_3=DenseAttribute(
                    units=3,
                    # dropout_rate=hp_custom["dense_3_dropout_rate"],
                    activation=hp_custom["dense_3_activation"],
                ),
            )
        elif self.model_name == "PZhouLSTMCNN":
            hp_custom = {}
            for key, value in self.model_config.items():
                hp_custom[value.get("name")] = self._get_hp_value(hp, value)
            custom_model = PZhouLSTMCNNModel(
                data_vocab_size=data_vocab_size,
                embedding_matrix=embedding_matrix,
                input_length=self.input_length,
                dropout_combine=hp_custom["dropout_combine"],
                dropout_features=hp_custom["dropout_features"],
                cnn_2d_attribute_1=Cnn2DAttribute(
                    filter_size=hp_custom["cnn_2d_1_filter_size"],
                    kernel_size=(
                        hp_custom["cnn_2d_1_kernel_size_height"],
                        hp_custom["cnn_2d_1_kernel_size_width"],
                    ),
                    padding=hp_custom["cnn_2d_1_padding"],
                    activation=hp_custom["cnn_2d_1_activation"],
                    # dropout_rate=hp_custom["cnn_2d_1_dropout_rate"],
                ),
                lstm_attributes_1=LSTMAttribute(
                    units=hp_custom["lstm_1_units"],
                    dropout_rate=hp_custom["lstm_1_dropout_rate"],
                ),
                dense_attributes_3=DenseAttribute(
                    units=3,
                    # dropout_rate=hp_custom["dense_3_dropout_rate"],
                    activation=hp_custom["dense_3_activation"],
                ),
            )
            
        else:  # Add this else block
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        custom_model.build_model()
        custom_model.compile_model(learning_rate=hp_custom["lr"])
        
        return custom_model.model
    
    # Định nghĩa hàm bất đồng bộ gửi file báo cáo qua Telegram
   

    def fit(self, hp, model, *args, **kwargs):  # Keep the fit method, but make it simpler
        # model.hp = hp # Store hp object inside model, so we can get hyperparameter value in on_train_begin

        # Tạo tên file với thời gian hiện tại
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"f1_score_report_{current_time}_{self.model_name}.txt"

        
        with open(report_filename, "w") as file:
            file.write(f"Model Name: {self.model_name}\n")
            file.write("Model Parameters:\n")
            if self.model_config is not None: # Check if not None
                for key, value in self.model_config.items():
                    if isinstance(value, dict) and "name" in value:
                        file.write(f"{value.get('name')}: {model.hp.get(value.get('name')) if hasattr(model, 'hp') else value}\n")
                    else:
                        file.write(f"{key}: {value}\n")

            file.write("\nWord2Vec Parameters:\n")
            if self.w2v_config is not None: # Check if not None
                for key, value in self.w2v_config.items():
                     if isinstance(value, dict) and "name" in value:
                        file.write(f"{value.get('name')}: {model.hp.get(value.get('name')) if hasattr(model, 'hp') else value}\n")
                     else:
                        file.write(f"{key}: {value}\n")
        
        # Đặt EarlyStopping callback
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True
        )
        # Huấn luyện mô hình
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epoch_num,
            batch_size=hp.get('batch_size'),
            callbacks=[early_stop],
            verbose=1
        )

        col_width = 10

        # Dự đoán trên tập validation và tính toán F1-score
        y_pred = model.predict(self.X_val, verbose=0)
        y_true_labels = np.argmax(self.y_val, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Tạo báo cáo classification với F1-score
        report = classification_report(y_true_labels, y_pred_labels, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0, digits=4)

        # Tính toán confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels)

        # Tạo chuỗi để ghi confusion matrix vào file
        cm_str = "\nConfusion Matrix On Validation Set:\n"
        cm_str += f"{'':<{col_width}}{'Negative':>{col_width}}{'Neutral':>{col_width}}{'Positive':>{col_width}}\n"
        cm_str += f"{'Negative':<{col_width}}{cm[0][0]:>{col_width}}{cm[0][1]:>{col_width}}{cm[0][2]:>{col_width}}\n"
        cm_str += f"{'Neutral':<{col_width}}{cm[1][0]:>{col_width}}{cm[1][1]:>{col_width}}{cm[1][2]:>{col_width}}\n"
        cm_str += f"{'Positive':<{col_width}}{cm[2][0]:>{col_width}}{cm[2][1]:>{col_width}}{cm[2][2]:>{col_width}}\n"


        y_pred_test = model.predict(self.X_test, verbose=0)
        y_true_labels_test = np.argmax(self.y_test, axis=1)
        y_pred_labels_test = np.argmax(y_pred_test, axis=1)

        # Tạo báo cáo classification với F1-score
        report_1 = classification_report(y_true_labels_test, y_pred_labels_test, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0, digits=4)

        # Tính toán confusion matrix
        cm_1 = confusion_matrix(y_true_labels_test, y_pred_labels_test)

        # Tạo chuỗi để ghi confusion matrix vào file
        # Độ rộng cột
        

        # Tạo chuỗi để ghi confusion matrix vào file
        cm_str_1 = "\nConfusion Matrix On Test Set:\n"
        cm_str_1 += f"{'':<{col_width}}{'Negative':>{col_width}}{'Neutral':>{col_width}}{'Positive':>{col_width}}\n"
        cm_str_1 += f"{'Negative':<{col_width}}{cm_1[0][0]:>{col_width}}{cm_1[0][1]:>{col_width}}{cm_1[0][2]:>{col_width}}\n"
        cm_str_1 += f"{'Neutral':<{col_width}}{cm_1[1][0]:>{col_width}}{cm_1[1][1]:>{col_width}}{cm_1[1][2]:>{col_width}}\n"
        cm_str_1 += f"{'Positive':<{col_width}}{cm_1[2][0]:>{col_width}}{cm_1[2][1]:>{col_width}}{cm_1[2][2]:>{col_width}}\n"


        # In báo cáo vào terminal
        print("\nClassification Report on Test Set:")
        print(report_1)

        # Lưu báo cáo vào file
        with open(report_filename, "a") as file:
            file.write("\nClassification Report On Validatation Set:\n")
            file.write(report)
            file.write("\nClassification Report On Test Set:\n")
            file.write(report_1)
            file.write(cm_str)
            file.write(cm_str_1)

        try:
            # Sử dụng asyncio.run() để chạy coroutine gửi file
            asyncio.run(send_report_via_telegram(report_filename, self.telegram_bot_id, self.group_chat_id))
        except Exception as e:
            print("Gửi báo cáo qua Telegram thất bại:", e)

        return history
