import multiprocessing
import keras_tuner as kt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from Model import CustomModel

class CustomHyperModel(kt.HyperModel):
    def __init__(self, w2v_corpus, tokenizer_data, input_length, X_train, y_train, X_val, y_val):
        super().__init__()
        self.w2v_corpus = w2v_corpus
        self.tokenizer_data = tokenizer_data
        self.input_length = input_length
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def build(self, hp):
        # Word2Vec Hyperparameters
        w2v_params = {
            'sg': hp.Choice('w2v_sg', [0, 1]),
            'vector_size': hp.Int('w2v_vector_size', 100, 300, step=100),
            'window': hp.Int('w2v_window', 3, 10, step=2),
            'min_count': hp.Int('w2v_min_count', 5, 20, step=5),
            'negative': hp.Int('w2v_negative', 5, 15, step=5),
            'sample': hp.Float('w2v_sample', 1e-5, 1e-3, sampling='log'),
            'epochs': 30  # Fixed for tuning efficiency
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

        # Build CustomModel
        custom_model = CustomModel(
            data_vocab_size=data_vocab_size,
            embedding_matrix=embedding_matrix,
            input_length=self.input_length
        )
        custom_model.build_model()

        # Tune CustomModel Hyperparameters
        hp_custom = {
            'learning_rate': hp.Float('lr', 1e-5, 1e-3, sampling='log'),
            'dropout': hp.Float('dropout', 0.1, 0.5, step=0.1),
            'dense_units': hp.Int('dense_units', 64, 256, step=64),
            'batch_size': hp.Choice('batch_size', [32, 64, 128])
        }

        custom_model.dropout_threshold = hp_custom['dropout']
        custom_model.compile_model(learning_rate=hp_custom['learning_rate'])
        
        return custom_model.model

    def fit(self, hp, model, *args, **kwargs):
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=hp.Int('patience', 3, 10),
            restore_best_weights=True
        )
        return model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=100,
            batch_size=hp.get('batch_size'),
            callbacks=[early_stop],
            verbose=1
        )