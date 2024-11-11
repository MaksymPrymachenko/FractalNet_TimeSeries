from torch import nn
import numpy as np
import torch

from .modules import Flatten, FractalBlock
from .base_model import BaseModelPL


class LSTM_Net(BaseModelPL):

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        input_size: int = 1,
        output_size: int = 1,
        bidirectional: bool = False,
        **kwargs
    ) -> None:
        super(LSTM_Net, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)

        out = self.fc(out[:, -1, :])

        return out


class LSTM_GRU_Net(BaseModelPL):

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        **kwargs
    ) -> None:
        super(LSTM_GRU_Net, self).__init__(**kwargs)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.lstm_dropout = nn.Dropout(0.1)

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        self.gru_dropout = nn.Dropout(0.1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)

        lstm_out = self.lstm_dropout(lstm_out)

        gru_out, _ = self.gru(lstm_out)

        gru_out = self.gru_dropout(gru_out)

        out = self.fc(gru_out[:, -1, :])
        return out


class CNN_LSTM_Net(BaseModelPL):

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        cnn_out_channels: int = 64,
        kernel_size: int = 4,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        super(CNN_LSTM_Net, self).__init__(**kwargs)

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.batch_norm = nn.BatchNorm1d(cnn_out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1).permute(0, 2, 1)

        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        out = self.fc(lstm_out[:, -1, :])

        return out


class FractalNet(BaseModelPL):

    def __init__(
        self,
        n_columns,
        init_channels,
        p_ldrop,
        dropout_probs,
        gdrop_ratio,
        input_size: int = 1,
        output_size: int = 1,
        gap=0,
        init="xavier",
        pad_type="zero",
        doubling=False,
        consist_gdrop=True,
        dropout_pos="CDBR",
        lstm_num_layers: int = 2,
        lstm_hidden_size: int = 256,
        **kwargs
    ):
        super(FractalNet, self).__init__(**kwargs)
        assert dropout_pos in ["CDBR", "CBRD", "FD"]

        dropout_probs = dropout_probs[:n_columns]

        self.B = len(dropout_probs)
        self.consist_gdrop = consist_gdrop
        self.gdrop_ratio = gdrop_ratio
        self.n_columns = n_columns
        C_in = 1

        layers = nn.ModuleList()
        C_out = init_channels
        total_layers = 0
        for b, p_dropout in enumerate(dropout_probs):
            print("[block {}] Channel in = {}, Channel out = {}".format(b, C_in, C_out))
            fb = FractalBlock(
                n_columns,
                C_in,
                C_out,
                p_ldrop,
                p_dropout,
                pad_type=pad_type,
                doubling=doubling,
                dropout_pos=dropout_pos,
            )
            layers.append(fb)
            if gap == 0 or b < self.B - 1:
                layers.append(nn.MaxPool1d(2))
            elif gap == 1:
                layers.append(nn.AdaptiveAvgPool1d(1))

            input_size = input_size // 2
            total_layers += fb.max_depth
            C_in = C_out
            if b < self.B - 2:
                C_out *= 2

        print("Last featuremap size = {}".format(input_size))
        print("Total layers = {}".format(total_layers))

        if gap == 2:
            layers.append(nn.Conv1d(C_out, output_size, 1, padding=0))
            layers.append(nn.AdaptiveAvgPool1d(1))
            layers.append(Flatten())
        else:
            # layers.append(Flatten())
            # layers.append(nn.Linear(C_out * size, n_classes))
            self.lstm = nn.LSTM(
                C_out,
                lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
                bidirectional=False,  # 512
            )
            self.fc = nn.Sequential(
                nn.Dropout(p_ldrop), nn.Linear(lstm_hidden_size, output_size)
            )  # (512, 1)

        self.layers = layers

        if init != "torch":
            initialize_ = {
                "xavier": nn.init.xavier_uniform_,
                "he": nn.init.kaiming_uniform_,
            }[init]

            for n, p in self.named_parameters():
                if p.dim() > 1:
                    initialize_(p)
                else:
                    if "bn.weight" in n:
                        nn.init.ones_(p)
                    else:
                        nn.init.zeros_(p)

    def forward(self, x, deepest=False):
        x = x.unsqueeze(1)

        if deepest:
            assert self.training is False
        GB = int(x.size(0) * self.gdrop_ratio)
        out = x
        global_cols = None
        for layer in self.layers:
            if isinstance(layer, FractalBlock):
                if not self.consist_gdrop or global_cols is None:
                    global_cols = np.random.randint(0, self.n_columns, size=[GB])

                out = layer(out, global_cols, deepest=deepest)
            else:
                out = layer(out)

        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])

        return out
