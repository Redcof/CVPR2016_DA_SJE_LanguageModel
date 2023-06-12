import torch
import torch.nn as nn

class HybridCNN(nn.Module):
    def __init__(self, alphasize, emb_dim, dropout=0.0, avg=0, cnn_dim=256):
        super(HybridCNN, self).__init__()

        self.dropout = dropout
        self.avg = avg
        self.cnn_dim = cnn_dim

        self.conv1 = nn.Conv1d(alphasize, 384, kernel_size=4)
        self.threshold1 = nn.Threshold()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv1d(384, 512, kernel_size=4)
        self.threshold2 = nn.Threshold()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv3 = nn.Conv1d(512, cnn_dim, kernel_size=4)
        self.threshold3 = nn.Threshold()
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.rnn = nn.RNN(cnn_dim, cnn_dim, batch_first=True)
        self.linear = nn.Linear(cnn_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape input from (batch_size, seq_len, alphasize) to (batch_size, alphasize, seq_len)

        x = self.pool1(self.threshold1(self.conv1(x)))
        x = self.pool2(self.threshold2(self.conv2(x)))
        x = self.pool3(self.threshold3(self.conv3(x)))

        x = x.permute(0, 2, 1)  # Reshape input back to (batch_size, seq_len, cnn_dim)

        _, hn = self.rnn(x)
        hn = hn.squeeze(0)  # Remove the batch dimension from hidden state

        out = self.linear(self.dropout_layer(hn))

        return out
