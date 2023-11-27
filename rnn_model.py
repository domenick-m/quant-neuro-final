import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, n_outputs=2):
        super().__init__()
        # stack of GRUs, take in shape: (B, T, N)
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          dropout=dropout, 
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        # pass through GRU
        x, _ = self.gru(x)
        # apply dropout
        x = self.dropout(x)
        # turn last layers outputs into behavior predictions
        x = self.readout(x)
        return x