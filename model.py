import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TNet(nn.Layer):
    def __init__(self, k=3):
        super(TNet, self).__init__()

        self.k = k
        self.mlps = nn.Sequential(
            nn.Linear(k, 64),
            nn.BatchNorm1D(64, data_format="NLC"),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1D(128, data_format="NLC"),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1D(1024, data_format="NLC"),
            nn.ReLU(),
        )
        self.max_pool = nn.AdaptiveMaxPool2D((1, 1024))
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

        self.const = paddle.eye(k, k, dtype=paddle.float32)

    def forward(self, x):
        x = self.mlps(x)

        x = paddle.unsqueeze(x, 0)
        x = self.max_pool(x)
        x = paddle.squeeze(x, 0)
        x = paddle.squeeze(x, -2)

        x = self.fc_layers(x)

        x = paddle.reshape(x, shape=(-1, self.k, self.k))
        x = x + self.const

        return x


class PointNet(nn.Layer):
    def __init__(self):
        super(PointNet, self).__init__()

        self.input_transform = TNet(k=3)
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1D(64, data_format="NLC"),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1D(64, data_format="NLC"),
            nn.ReLU(),
        )
        self.feature_transform = TNet(k=64)
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1D(64, data_format="NLC"),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1D(128, data_format="NLC"),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1D(1024, data_format="NLC"),
            nn.ReLU(),
        )
        self.max_pool = nn.AdaptiveMaxPool2D((1, 1024))
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(256, 40),
        )

    def forward(self, x):
        batch_size, num_point, _ = x.shape
        x = x @ self.input_transform(x)
        x = self.mlp1(x)
        x = x @ self.feature_transform(x)
        x = self.mlp2(x)
        x = paddle.unsqueeze(x, 0)
        x = self.max_pool(x)
        x = paddle.squeeze(x, 0)
        x = paddle.squeeze(x, -2)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    x = paddle.randn((8, 512, 3))
    model = PointNet("", "")
    out = model(x)
    print(out.shape)
