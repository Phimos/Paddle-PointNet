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

        x = paddle.transpose(x, (0, 2, 1))
        x = F.adaptive_avg_pool1d(x, 1)
        x = paddle.squeeze(x, -1)

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
            nn.Dropout(p=0.7),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Linear(256, 40),
        )

    def forward(self, x):
        batch_size, num_point, _ = x.shape
        x = x @ self.input_transform(x)
        x = self.mlp1(x)
        x = x @ self.feature_transform(x)
        x = self.mlp2(x)
        x = paddle.transpose(x, (0, 2, 1))
        x = F.adaptive_avg_pool1d(x, 1)
        x = paddle.squeeze(x, -1)
        x = self.fc_layers(x)
        return x


class STN3d(nn.Layer):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1D(channel, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.iden = paddle.eye(3, 3, dtype=paddle.float32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.reshape((-1, 3, 3)) + self.iden
        return x


class STNkd(nn.Layer):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1D(k, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.k = k
        self.iden = paddle.eye(self.k, self.k, dtype=paddle.float32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.reshape((-1, self.k, self.k)) + self.iden
        return x


class PointNetEncoder(nn.Layer):
    def __init__(
        self, global_feat=True, input_transform=True, feature_transform=False, channel=3
    ):
        super(PointNetEncoder, self).__init__()

        self.global_feat = global_feat
        if input_transform:
            self.input_transfrom = STNkd(k=channel)
        else:
            self.input_transfrom = lambda x: paddle.eye(
                channel, channel, dtype=paddle.float32
            )

        self.conv1 = nn.Conv1D(channel, 64, 1)
        self.conv2 = nn.Conv1D(64, 64, 1)
        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(64)

        if feature_transform:
            self.feature_transform = STNkd(k=64)
        else:
            self.feature_transform = lambda x: paddle.eye(64, 64, dtype=paddle.float32)

        self.conv3 = nn.Conv1D(64, 64, 1)
        self.conv4 = nn.Conv1D(64, 128, 1)
        self.conv5 = nn.Conv1D(128, 1024, 1)
        self.bn3 = nn.BatchNorm1D(64)
        self.bn4 = nn.BatchNorm1D(128)
        self.bn5 = nn.BatchNorm1D(1024)

    def forward(self, x):
        x = paddle.transpose(x, (0, 2, 1))
        B, D, N = x.shape
        trans_input = self.input_transfrom(x)
        x = paddle.transpose(x, (0, 2, 1))
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = paddle.bmm(x, trans_input)
        if D > 3:
            x = paddle.cat([x, feature], dim=2)
        x = paddle.transpose(x, (0, 2, 1))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        trans_feat = self.feature_transform(x)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.bmm(x, trans_feat)
        x = paddle.transpose(x, (0, 2, 1))

        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        if self.global_feat:
            return x, trans_input, trans_feat
        else:
            x = x.reshape((-1, 1024, 1)).repeat(1, 1, N)
            return paddle.cat([x, pointfeat], 1), trans_input, trans_feat


class PointNetClassifier(nn.Layer):
    def __init__(self, k=40, normal_channel=False):
        super(PointNetClassifier, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(
            global_feat=True,
            input_transform=True,
            feature_transform=True,
            channel=channel,
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1D(512)
        self.bn2 = nn.BatchNorm1D(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans_input, trans_feat = self.feat(x)
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_input, trans_feat


class CrossEntropyMatrixRegularization(nn.Layer):
    def __init__(self, mat_diff_loss_scale=1e-3):
        super(CrossEntropyMatrixRegularization, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat=None):
        loss = F.cross_entropy(pred, target)

        if trans_feat is None:
            mat_diff_loss = 0
        else:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


def feature_transform_reguliarzer(trans):
    d = trans.shape[1]
    I = paddle.eye(d)
    loss = paddle.mean(
        paddle.norm(
            paddle.bmm(trans, paddle.transpose(trans, (0, 2, 1))) - I, axis=(1, 2)
        )
    )
    return loss
