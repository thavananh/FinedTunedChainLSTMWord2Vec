import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, data_vocab_size, embedding_matrix, dropout_threshold=0.1):
        super(CNNLSTMAttentionModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=data_vocab_size, embedding_dim=300)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embedding layer

        self.dropout = nn.Dropout(0.5)
        self.conv1d = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, padding='same')
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_threshold)

        self.bi_lstm = nn.LSTM(input_size=100, hidden_size=300, bidirectional=True, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(600)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=600, num_heads=12)
        self.layer_norm2 = nn.LayerNorm(600)
        self.dropout2 = nn.Dropout(dropout_threshold)

        self.global_maxpool1d = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(700, 256)
        self.dropout3 = nn.Dropout(dropout_threshold)
        self.fc2 = nn.Linear(256, 64)
        self.dropout4 = nn.Dropout(dropout_threshold)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        # Convolutional Path
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, embedding_dim, sequence_length)
        cnn_feature = self.conv1d(x)
        cnn_feature = self.maxpool1d(cnn_feature)
        cnn_feature = self.dropout1(cnn_feature)

        # Recurrent Path
        cnn_feature = cnn_feature.permute(0, 2, 1)  # Change shape back to (batch_size, sequence_length, embedding_dim)
        bi_lstm_feature, _ = self.bi_lstm(cnn_feature)
        bi_lstm_feature = self.maxpool1d(bi_lstm_feature.permute(0, 2, 1)).permute(0, 2, 1)
        bi_lstm_feature = self.layer_norm1(bi_lstm_feature)

        # Self-Attention Layer
        attention_output, _ = self.multihead_attn(bi_lstm_feature, bi_lstm_feature, bi_lstm_feature)
        attention_output = self.layer_norm2(attention_output)
        attention_output = self.dropout2(attention_output)

        # Apply GlobalMaxPooling1D to both feature maps
        cnn_pooled = self.global_maxpool1d(cnn_feature.permute(0, 2, 1)).squeeze(2)
        attention_pooled = self.global_maxpool1d(attention_output.permute(0, 2, 1)).squeeze(2)

        # Concatenate the pooled features
        combine_feature = torch.cat((cnn_pooled, attention_pooled), dim=1)
        combine_feature = self.layer_norm1(combine_feature)
        combine_feature = self.dropout1(combine_feature)

        # Classification Layers
        classifier = torch.relu(self.fc1(combine_feature))
        classifier = self.dropout3(classifier)
        classifier = torch.relu(self.fc2(classifier))
        classifier = self.dropout4(classifier)
        classifier = torch.softmax(self.fc3(classifier), dim=1)

        return classifier

    def train_model(self, train_features, train_labels, dev_features, dev_labels, epochs=500, batch_size=64, patience=50):
        train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.long), torch.tensor(train_labels, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        dev_dataset = TensorDataset(torch.tensor(dev_features, dtype=torch.long), torch.tensor(dev_labels, dtype=torch.long))
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)

        early_stopping_counter = 0
        best_val_accuracy = 0

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            self.eval()
            val_accuracy = 0
            with torch.no_grad():
                for batch in dev_loader:
                    inputs, labels = batch
                    outputs = self(inputs)
                    val_accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()
            val_accuracy /= len(dev_dataset)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print("Early stopping")
                break

    def evaluate_model(self, test_features, test_labels, label_class):
        self.eval()
        with torch.no_grad():
            test_outputs = self(torch.tensor(test_features, dtype=torch.long))
            preds = torch.argmax(test_outputs, dim=1).numpy()

        print(classification_report(np.argmax(test_labels, axis=1), preds, target_names=label_class, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(np.argmax(test_labels, axis=1), preds, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_class)
        disp.plot()
        plt.gca().grid(False)
        plt.title('Bi-LSTM with Multi Head Attention')
        plt.show()

# Example usage
data_vocab_size = 10000  # Example vocabulary size
embedding_matrix = np.random.randn(data_vocab_size, 300)  # Example embedding matrix

model = CNNLSTMAttentionModel(data_vocab_size, embedding_matrix)

# Assuming you have your data ready
# train_features, train_labels, dev_features, dev_labels, test_features, test_labels

# Train the model
model.train_model(train_features, train_labels, dev_features, dev_labels)

# Evaluate the model
label_class = ['Positive', 'Neutral', 'Negative']
model.evaluate_model(test_features, test_labels, label_class)