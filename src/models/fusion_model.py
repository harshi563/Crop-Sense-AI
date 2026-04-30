"""
models/fusion_model.py
CNN (satellite) + BiLSTM (weather) + MLP (soil) → Fusion → yield
"""
import torch
import torch.nn as nn


class SatCNN(nn.Module):
    def __init__(self, in_ch=6, out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,32,3,padding=1),    nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32,64,3,padding=1),    nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,128,3,padding=1),   nn.BatchNorm2d(128),nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256, out),     nn.ReLU(True),
        )
    def forward(self, x): return self.head(self.net(x))


class WxLSTM(nn.Module):
    def __init__(self, in_f=9, hidden=128, layers=2, out=128, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_f, hidden, layers, batch_first=True,
                            dropout=drop if layers>1 else 0, bidirectional=True)
        self.attn = nn.Linear(hidden*2, 1)
        self.head = nn.Sequential(nn.Linear(hidden*2, out), nn.ReLU(True), nn.Dropout(drop))

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (w * out).sum(1)
        return self.head(ctx)


class SoilMLP(nn.Module):
    def __init__(self, in_f, out=64, drop=0.2):
        super().__init__()
        self.bn  = nn.BatchNorm1d(in_f)
        self.fc1 = nn.Sequential(nn.Linear(in_f,128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(drop))
        self.fc2 = nn.Sequential(nn.Linear(128,128),  nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(drop))
        self.skip= nn.Linear(in_f, 128)
        self.head= nn.Sequential(nn.Linear(128, out), nn.ReLU(True))

    def forward(self, x):
        x = self.bn(x)
        return self.head(self.fc2(self.fc1(x)) + self.skip(x))


class FusionModel(nn.Module):
    def __init__(self, sat_ch=6, wx_f=9, soil_f=16, drop=0.3):
        super().__init__()
        self.cnn  = SatCNN(sat_ch, 64)
        self.lstm = WxLSTM(wx_f, 128, 2, 128, drop)
        self.mlp  = SoilMLP(soil_f, 64, drop)
        total = 64 + 128 + 64   # 256
        self.fusion = nn.Sequential(
            nn.Linear(total, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(drop*0.7),
            nn.Linear(128, 64),    nn.ReLU(True), nn.Dropout(drop*0.5),
            nn.Linear(64, 1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, sat, wx, soil):
        return self.fusion(torch.cat([self.cnn(sat), self.lstm(wx), self.mlp(soil)], dim=1))
