import torch
import torch.nn as nn



class RNNMODEL(nn.Module):
    def __init__(
        self, 
        input_size = None, 
        hidden_size=None, 
        num_layers=None, 
        readout_size=2, 
        dropout = 0.1, 
        gru_kwargs={}, 
        readout_kwargs={}
    ):
        super(RNNMODEL,self).__init__()
        params = [input_size, hidden_size, num_layers]
        if None in params:
            raise Exception
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.readout_size = readout_size
        self.gru_kwargs = gru_kwargs
        self.readout_kwargs = readout_kwargs
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.norms = nn.ModuleList()
        self.grus = nn.ModuleList()
        for _ in range(self.num_layers):
            self.grus.append(nn.GRU(input_size=input_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=1, 
                                    dropout=dropout, 
                                    **gru_kwargs))
            input_size = hidden_size
            # self.norms.append(nn.BatchNorm1d(hidden_size))
            self.norms.append(nn.LayerNorm(hidden_size))
        self.readout = nn.Linear(in_features=self.hidden_size, 
                                 out_features=self.readout_size)


    def forward(self, x):
        for i in range(self.num_layers):
            x, _ = self.grus[i](x)
            # x = self.norms[i](x)
        readout_out = self.readout(x)
        return readout_out


    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()  # Set the model to evaluation mode if needed


