from datetime import datetime
import multiprocessing
import keras_tuner as kt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import tensorflow as tf
from Model import CustomModel

import yaml


class CustomHyperModel(kt.HyperModel):
    def __init__(self, w2v_corpus, tokenizer_data, input_length, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
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
        self.strategy = tf.distribute.MirroredStrategy()
        self.model_name = model_name

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    
    def build(self, hp):
        # Word2Vecc Hyperparameters
        # with self.strategy.scope():
        config = self.load_config('config.yaml')
        custom_model_config = config.get(self.model_name, {})
        w2v_params = {
            'sg': hp.Choice('w2v_sg', [1]),
            'vector_size': hp.Int('w2v_vector_size', 100, 300, step=100),
            'window': hp.Int('w2v_window', 3, 10, step=2),
            'min_count': hp.Int('w2v_min_count', 2, 20, step=2),
            'negative': hp.Int('w2v_negative', 5, 15, step=5),
            'sample': hp.Float('w2v_sample', 1e-5, 1e-3, sampling='log'),
            'epochs': 30  
        }

        # Train Word2Vec
        w2v_model = Word2Vec(
            **w2v_params,
            workers=multiprocessing.cpu_count()
        )
        w2v_model.build_vocab(self.w2v_corpus)
        w2v_model.train(
            self.w2v_corpus, 
            total_examples=len(self.w2v_corpus), 
            epochs=w2v_params['epochs']
        )

        # Generate Embedding Matrix
        data_vocab_size = len(self.tokenizer_data.word_index) + 1
        embedding_matrix = np.random.normal(
            0, 0.05, 
            (data_vocab_size, w2v_params['vector_size'])
        )
        
        for word, i in self.tokenizer_data.word_index.items():
            if i >= data_vocab_size:
                continue
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]

        if self.model_name == 'model':
            custom_model = CustomModel(
                data_vocab_size=data_vocab_size,
                embedding_matrix=embedding_matrix,
                input_length=self.input_length
            )
            custom_model.build_model()

            # Tune CustomModel Hyperparameters
            hp_custom = {
                # General hyperparameters
                'learning_rate': hp.Float('lr', 1e-4, 1e-3, sampling='log'),
                'batch_size': hp.Choice('batch_size', [32, 64, 128]),
                
                # Dropout-related hyperparameters
                'dropout_features': hp.Float('dropout_features', 0.0, 0.5, step=0.1),
                'dropout_combine': hp.Float('dropout_combine', 0.0, 0.5, step=0.1),

                # CNN-related hyperparameters
                'cnn_1_filter_size': hp.Int('cnn_1_filter_size', 32, 256, step=32),
                'cnn_1_kernel_size': hp.Int('cnn_1_kernel_size', 1, 4, step=1),
                'cnn_1_padding': hp.Choice('cnn_1_padding', ['valid', 'same']),
                # 'cnn_1_activation': hp.Choice('cnn_1_activation', ['relu', 'elu', 'gelu','silu']),
                'cnn_1_activation': hp.Choice('cnn_1_activation', ['silu']),
                'cnn_1_dropout_rate': hp.Float('cnn_1_dropout_rate', 0.0, 0.5, step=0.1),

                'cnn_2_filter_size': hp.Int('cnn_2_filter_size', 32, 256, step=32),
                'cnn_2_kernel_size': hp.Int('cnn_2_kernel_size', 1, 4, step=1),
                'cnn_2_padding': hp.Choice('cnn_2_padding', ['valid', 'same']),
                # 'cnn_2_activation': hp.Choice('cnn_2_activation', ['relu', 'elu', 'gelu','silu']),
                'cnn_2_activation': hp.Choice('cnn_2_activation', ['silu']),
                'cnn_2_dropout_rate': hp.Float('cnn_2_dropout_rate', 0.0, 0.5, step=0.1),

                'cnn_3_filter_size': hp.Int('cnn_3_filter_size', 32, 256, step=32),
                'cnn_3_kernel_size': hp.Int('cnn_3_kernel_size', 1, 4, step=1),
                'cnn_3_padding': hp.Choice('cnn_3_padding', ['valid', 'same']),
                # 'cnn_3_activation': hp.Choice('cnn_3_activation', ['relu', 'elu', 'gelu','silu']),
                'cnn_3_activation': hp.Choice('cnn_3_activation', ['silu']),
                'cnn_3_dropout_rate': hp.Float('cnn_3_dropout_rate', 0.0, 0.5, step=0.1),

                'cnn_4_filter_size': hp.Int('cnn_4_filter_size', 32, 256, step=32),
                'cnn_4_kernel_size': hp.Int('cnn_4_kernel_size', 1, 4, step=1),
                'cnn_4_padding': hp.Choice('cnn_4_padding', ['valid', 'same']),
                # 'cnn_4_activation': hp.Choice('cnn_4_activation', ['relu', 'elu', 'gelu','silu']),
                'cnn_4_activation': hp.Choice('cnn_4_activation', ['silu']),
                'cnn_4_dropout_rate': hp.Float('cnn_4_dropout_rate', 0.0, 0.5, step=0.1),

                # LSTM-related hyperparameters
                'lstm_1_units': hp.Int('lstm_1_units', 64, 512, step=64),
                'lstm_1_dropout_rate': hp.Float('lstm_1_dropout_rate', 0.0, 0.5, step=0.1),

                'lstm_2_units': hp.Int('lstm_2_units', 64, 512, step=64),
                'lstm_2_dropout_rate': hp.Float('lstm_2_dropout_rate', 0.0, 0.5, step=0.1),

                # Multi-head Attention-related hyperparameters
                'multi_head_attention_num_heads': hp.Int('multi_head_attention_num_heads', 4, 16, step=4),
                'multi_head_attention_key_dim': hp.Int('multi_head_attention_key_dim', 32, 128, step=32),
                'multi_head_attention_dropout_rate': hp.Float('multi_head_attention_dropout_rate', 0.0, 0.5, step=0.1),

                # Dense-related hyperparameters
                'dense_1_units': hp.Int('dense_1_units', 64, 512, step=64),
                'dense_1_dropout_rate': hp.Float('dense_1_dropout_rate', 0.0, 0.5, step=0.1),
                # 'dense_1_activation': hp.Choice('dense_1_activation', ['relu', 'elu', 'gelu','silu']),
                'dense_1_activation': hp.Choice('dense_1_activation', ['silu']),

                'dense_2_units': hp.Int('dense_2_units', 64, 512, step=64),
                'dense_2_dropout_rate': hp.Float('dense_2_dropout_rate', 0.0, 0.5, step=0.1),
                # 'dense_2_activation': hp.Choice('dense_2_activation', ['relu', 'elu', 'gelu','silu']),
                'dense_2_activation': hp.Choice('dense_2_activation', ['silu']),

                'dense_3_units': hp.Int('dense_3_units', 64, 512, step=64),
                'dense_3_dropout_rate': hp.Float('dense_3_dropout_rate', 0.0, 0.5, step=0.1),
                'dense_3_activation': hp.Choice('dense_3_activation', ['softmax', 'log_softmax']),
            }

            custom_model.dropout_combine = hp_custom['dropout_combine']
            custom_model.dropout_features = hp_custom['dropout_features']
            custom_model.cnn_1_filter_size = hp_custom['cnn_1_filter_size']
            custom_model.cnn_1_kernel_size = hp_custom['cnn_1_kernel_size']
            custom_model.cnn_1_padding = hp_custom['cnn_1_padding']
            custom_model.cnn_1_activation = hp_custom['cnn_1_activation']
            custom_model.cnn_1_dropout_rate = hp_custom['cnn_1_dropout_rate']
            custom_model.cnn_2_filter_size = hp_custom['cnn_2_filter_size']
            custom_model.cnn_2_kernel_size = hp_custom['cnn_2_kernel_size']
            custom_model.cnn_2_padding = hp_custom['cnn_2_padding']
            custom_model.cnn_2_activation = hp_custom['cnn_2_activation']
            custom_model.cnn_2_dropout_rate = hp_custom['cnn_2_dropout_rate']
            custom_model.cnn_3_filter_size = hp_custom['cnn_3_filter_size']
            custom_model.cnn_3_kernel_size = hp_custom['cnn_3_kernel_size']
            custom_model.cnn_3_padding = hp_custom['cnn_3_padding']
            custom_model.cnn_3_activation = hp_custom['cnn_3_activation']
            custom_model.cnn_3_dropout_rate = hp_custom['cnn_3_dropout_rate']
            custom_model.cnn_4_filter_size = hp_custom['cnn_4_filter_size']
            custom_model.cnn_4_kernel_size = hp_custom['cnn_4_kernel_size']
            custom_model.cnn_4_padding = hp_custom['cnn_4_padding']
            custom_model.cnn_4_activation = hp_custom['cnn_4_activation']
            custom_model.cnn_4_dropout_rate = hp_custom['cnn_4_dropout_rate']
            custom_model.lstm_1_units = hp_custom['lstm_1_units']
            custom_model.lstm_1_dropout_rate = hp_custom['lstm_1_dropout_rate']
            custom_model.lstm_2_units = hp_custom['lstm_2_units']
            custom_model.lstm_2_dropout_rate = hp_custom['lstm_2_dropout_rate']
            custom_model.multi_head_attention_num_heads = hp_custom['multi_head_attention_num_heads']
            custom_model.multi_head_attention_key_dim = hp_custom['multi_head_attention_key_dim']
            custom_model.multi_head_attention_dropout_rate = hp_custom['multi_head_attention_dropout_rate']
            custom_model.dense_1_units = hp_custom['dense_1_units']
            custom_model.dense_1_dropout_rate = hp_custom['dense_1_dropout_rate']
            custom_model.dense_1_activation = hp_custom['dense_1_activation']
            custom_model.dense_2_units = hp_custom['dense_2_units']
            custom_model.dense_2_dropout_rate = hp_custom['dense_2_dropout_rate']
            custom_model.dense_2_activation = hp_custom['dense_2_activation']
            custom_model.dense_3_units = hp_custom['dense_3_units']
            custom_model.dense_3_dropout_rate = hp_custom['dense_3_dropout_rate']
            custom_model.dense_3_activation = hp_custom['dense_3_activation']
            custom_model.compile_model(learning_rate=hp_custom['learning_rate'])
            
        return custom_model.model

    from sklearn.metrics import confusion_matrix

    def fit(self, hp, model, *args, **kwargs):
        # Lấy thông tin về các tham số của mô hình và Word2Vec
        # with self.strategy.scope():
        model_params = {
            # General hyperparameters
            'learning_rate': hp.get('lr'),
            'batch_size': hp.get('batch_size'),
            
            # Dropout-related hyperparameters
            'dropout_features': hp.get('dropout_features'),
            'dropout_combine': hp.get('dropout_combine'),

            # CNN-related hyperparameters
            'cnn_1_filter_size': hp.get('cnn_1_filter_size'),
            'cnn_1_kernel_size': hp.get('cnn_1_kernel_size'),
            'cnn_1_padding': hp.get('cnn_1_padding'),
            'cnn_1_activation': hp.get('cnn_1_activation'),
            'cnn_1_dropout_rate': hp.get('cnn_1_dropout_rate'),

            'cnn_2_filter_size': hp.get('cnn_2_filter_size'),
            'cnn_2_kernel_size': hp.get('cnn_2_kernel_size'),
            'cnn_2_padding': hp.get('cnn_2_padding'),
            'cnn_2_activation': hp.get('cnn_2_activation'),
            'cnn_2_dropout_rate': hp.get('cnn_2_dropout_rate'),

            'cnn_3_filter_size': hp.get('cnn_3_filter_size'),
            'cnn_3_kernel_size': hp.get('cnn_3_kernel_size'),
            'cnn_3_padding': hp.get('cnn_3_padding'),
            'cnn_3_activation': hp.get('cnn_3_activation'),
            'cnn_3_dropout_rate': hp.get('cnn_3_dropout_rate'),

            'cnn_4_filter_size': hp.get('cnn_4_filter_size'),
            'cnn_4_kernel_size': hp.get('cnn_4_kernel_size'),
            'cnn_4_padding': hp.get('cnn_4_padding'),
            'cnn_4_activation': hp.get('cnn_4_activation'),
            'cnn_4_dropout_rate': hp.get('cnn_4_dropout_rate'),

            # LSTM-related hyperparameters
            'lstm_1_units': hp.get('lstm_1_units'),
            'lstm_1_dropout_rate': hp.get('lstm_1_dropout_rate'),

            'lstm_2_units': hp.get('lstm_2_units'),
            'lstm_2_dropout_rate': hp.get('lstm_2_dropout_rate'),

            # Multi-head Attention-related hyperparameters
            'multi_head_attention_num_heads': hp.get('multi_head_attention_num_heads'),
            'multi_head_attention_key_dim': hp.get('multi_head_attention_key_dim'),
            'multi_head_attention_dropout_rate': hp.get('multi_head_attention_dropout_rate'),

            # Dense-related hyperparameters
            'dense_1_units': hp.get('dense_1_units'),
            'dense_1_dropout_rate': hp.get('dense_1_dropout_rate'),
            'dense_1_activation': hp.get('dense_1_activation'),

            'dense_2_units': hp.get('dense_2_units'),
            'dense_2_dropout_rate': hp.get('dense_2_dropout_rate'),
            'dense_2_activation': hp.get('dense_2_activation'),

            'dense_3_units': hp.get('dense_3_units'),
            'dense_3_dropout_rate': hp.get('dense_3_dropout_rate'),
            'dense_3_activation': hp.get('dense_3_activation'),
        }

        w2v_params = {
            'w2v_sg': hp.get('w2v_sg'),
            'w2v_vector_size': hp.get('w2v_vector_size'),
            'w2v_window': hp.get('w2v_window'),
            'w2v_min_count': hp.get('w2v_min_count'),
            'w2v_negative': hp.get('w2v_negative'),
            'w2v_sample': hp.get('w2v_sample'),
        }

        # Tạo tên file với thời gian hiện tại
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"f1_score_report_{current_time}.txt"

        # Lưu thông tin tham số vào file
        with open(report_filename, "w") as file:
            file.write("Model Parameters:\n")
            for key, value in model_params.items():
                file.write(f"{key}: {value}\n")

            file.write("\nWord2Vec Parameters:\n")
            for key, value in w2v_params.items():
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
            epochs=100,
            batch_size=hp.get('batch_size'),
            callbacks=[early_stop],
            verbose=1
        )

        # Dự đoán trên tập validation và tính toán F1-score
        y_pred = model.predict(self.X_val, verbose=0)
        y_true_labels = np.argmax(self.y_val, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Tạo báo cáo classification với F1-score
        report = classification_report(y_true_labels, y_pred_labels, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0, digits=4)

        # Tính toán confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels)

        # Tạo chuỗi để ghi confusion matrix vào file
        cm_str = "Confusion Matrix On Validation Set:\n"
        cm_str += "    Negative  Neutral  Positive\n"
        cm_str += f"Negative   {cm[0][0]}      {cm[0][1]}      {cm[0][2]}\n"
        cm_str += f"Neutral    {cm[1][0]}      {cm[1][1]}      {cm[1][2]}\n"
        cm_str += f"Positive   {cm[2][0]}      {cm[2][1]}      {cm[2][2]}\n"

        print(cm_str)

        # In báo cáo vào terminal
        print("\nClassification Report on Validation Set:")
        print(report)

        # Lưu báo cáo vào file
        with open(report_filename, "a") as file:
            file.write("\nClassification Report:\n")
            file.write(report)
            file.write("\n")
            file.write(cm_str)


        y_pred_test = model.predict(self.X_test, verbose=0)
        y_true_labels_test = np.argmax(self.y_test, axis=1)
        y_pred_labels_test = np.argmax(y_pred_test, axis=1)

        # Tạo báo cáo classification với F1-score
        report_1 = classification_report(y_true_labels_test, y_pred_labels_test, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0, digits=4)

        # Tính toán confusion matrix
        cm_1 = confusion_matrix(y_true_labels_test, y_pred_labels_test)

        # Tạo chuỗi để ghi confusion matrix vào file
        cm_str_1 = "Confusion Matrix On Test Set:\n"
        cm_str_1 += "    Negative  Neutral  Positive\n"
        cm_str_1 += f"Negative   {cm_1[0][0]}      {cm_1[0][1]}      {cm_1[0][2]}\n"
        cm_str_1 += f"Neutral    {cm_1[1][0]}      {cm_1[1][1]}      {cm_1[1][2]}\n"
        cm_str_1 += f"Positive   {cm_1[2][0]}      {cm_1[2][1]}      {cm_1[2][2]}\n"

        # In báo cáo vào terminal
        print("\nClassification Report on Test Set:")
        print(report_1)

        # Lưu báo cáo vào file
        with open(report_filename, "a") as file:
            file.write("\nClassification Report On Validatation Set:\n")
            file.write(report)
            file.write("\nClassification Report On Test Set:\n")
            file.write(report_1)
            file.write("\nConfusion Matrix On Validatation Set:\n")
            file.write(cm_str)
            file.write("\nConfusion Matrix On Validatation Set:\n")
            file.write(cm_str_1)

        return history
