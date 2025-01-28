from regex import B
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPool1D, Dropout, LayerNormalization, Bidirectional, LSTM, Concatenate, GlobalMaxPooling1D, Dense, MultiHeadAttention, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

class CustomModel:
    def __init__(self, data_vocab_size, embedding_matrix, input_length=110):
        self.data_vocab_size = data_vocab_size
        self.embedding_matrix = embedding_matrix
        self.input_length = input_length
        self.model = None
        self.history = None
        
        # Model hyperparameters
        self.dropout_threshold = 0.2
        self.embedding_output_dim = 300
        self.initializer = tf.keras.initializers.GlorotNormal()
    
    def build_model(self):
        input_layer = Input(shape=(self.input_length,))
        
        # Embedding layer
        x = Embedding(
            input_dim=self.data_vocab_size, 
            output_dim=self.embedding_output_dim,
            embeddings_initializer=self.initializer,
            weights=[self.embedding_matrix]
        )(input_layer)
        x = Dropout(0.5)(x)
        
        # Convolutional block
        cnn = Conv1D(100, 3, padding='same', activation='relu')(x)
        cnn = MaxPool1D()(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(self.dropout_threshold)(cnn)
        cnn = Conv1D(200, 3, padding='same', activation='relu')(x)
        cnn = MaxPool1D()(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(self.dropout_threshold)(cnn)
        
        # Bidirectional LSTM
        lstm = Bidirectional(LSTM(300, return_sequences=True, 
                                kernel_initializer=self.initializer))(cnn)
        lstm = MaxPool1D()(lstm)
        lstm = LayerNormalization()(lstm)
        lstm = Dropout(self.dropout_threshold)(lstm)
        # Multi-head Attention
        attention = MultiHeadAttention(num_heads=12, key_dim=32)(lstm, lstm)
        attention = LayerNormalization()(attention)
        attention = Dropout(self.dropout_threshold)(attention)
        
        # Feature pooling and concatenation
        cnn_pool = GlobalMaxPooling1D()(cnn)
        attn_pool = GlobalMaxPooling1D()(attention)
        combined = Concatenate()([cnn_pool, attn_pool])
        combined = LayerNormalization()(combined)
        combined = Dropout(self.dropout_threshold)(combined)
        
        # Classification head
        dense = Dense(256, activation='relu')(combined)
        dense = Dropout(self.dropout_threshold)(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(self.dropout_threshold)(dense)
        output = Dense(3, activation='softmax')(dense)
        
        self.model = Model(inputs=input_layer, outputs=output)
    
    def compile_model(self, learning_rate=1e-4, weight_decay=0.0):
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, 
             epochs=100, batch_size=64, patience=50):
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True
        )
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
    
    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0)
    
    def generate_classification_report(self, y_true, y_pred, labels=['Negative', 'Neutral', 'Positive']):
        print(y_true)
        print(y_pred)
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        print(classification_report(
            y_true_labels, y_pred_labels,
            target_names=labels,
            zero_division=0
        ))
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=['Negative', 'Neutral', 'Positive']):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=labels
        )
        disp.plot(cmap='Blues')
        plt.grid(False)
        plt.title('Bi-LSTM with Multi-Head Attention')
        plt.show()
