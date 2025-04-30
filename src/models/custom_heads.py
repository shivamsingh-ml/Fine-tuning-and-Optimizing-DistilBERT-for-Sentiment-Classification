import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2)
        )

    def forward(self, x):
        return self.head(x)


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation="relu", dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(activation, nn.ReLU())

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 2))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
