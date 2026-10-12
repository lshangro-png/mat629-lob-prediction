"""The three sequence classifiers. Input (B, T, F) → logits (B, 3)."""
import math

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# 1-D residual CNN
# ══════════════════════════════════════════════════════════════════════════════
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.dropout(self.act(out + self.skip(x)))


class LOBCNN(nn.Module):
    def __init__(self, n_features, n_filters=64, kernel_size=3, n_blocks=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, n_filters, kernel_size=1),
            nn.BatchNorm1d(n_filters), nn.GELU())
        blocks, ch = [], n_filters
        for i in range(n_blocks):
            out_ch = min(n_filters * 2 ** i, n_filters * 4)
            blocks.append(ResidualBlock1D(ch, out_ch, kernel_size, dropout))
            ch = out_ch
        self.res_blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(ch, ch // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(ch // 2, 3))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.res_blocks(self.input_proj(x.permute(0, 2, 1)))   # (B, C, T)
        return self.head(x[:, :, -1])                               # last time step


# ══════════════════════════════════════════════════════════════════════════════
# Bidirectional LSTM
# ══════════════════════════════════════════════════════════════════════════════
class LOBLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=32, n_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_size, 3))
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.size(0)
                with torch.no_grad():
                    p[n // 4: n // 2].fill_(1.0)   # forget-gate bias
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)   # last layer, forward + backward
        return self.head(self.norm(h))


# ══════════════════════════════════════════════════════════════════════════════
# Transformer encoder
# ══════════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class LOBTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model // 2, 3))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.encoder(self.pos_enc(self.proj(x)))
        return self.head(h[:, -1, :])   # last token


MODELS = {"cnn": LOBCNN, "lstm": LOBLSTM, "transformer": LOBTransformer}
