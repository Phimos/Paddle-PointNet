import argparse

import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.metric import Accuracy
from paddle.optimizer import Adam
from paddle.optimizer.lr import StepDecay

from ddata import ModelNetDataset
from model import CrossEntropyMatrixRegularization, PointNetClassifier


def parse_args():
    parser = argparse.ArgumentParser("Train")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size in training"
    )
    parser.add_argument("--num_category", type=int, default=40, help="ModelNet10/40")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate in training"
    )
    parser.add_argument("--num_point", type=int, default=1024, help="point number")
    parser.add_argument("--max_epochs", type=int, default=200, help="max epochs")
    parser.add_argument("--num_workers", type=int, default=32, help="num wrokers")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="pointnet.pdparams")
    # parser.add_argument("--data_dir", type=str, default="/data3/ganyunchong/ModelNet40")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data3/ganyunchong/Pointnet_Pointnet2_pytorch-master/data/modelnet40_normal_resampled",
    )

    return parser.parse_args()


def test(args):

    test_data = ModelNetDataset(args.data_dir, split="test", num_point=args.num_point)
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    model = PointNetClassifier()

    loss_fn = CrossEntropyMatrixRegularization()
    metrics = Accuracy()

    model_state_dict = paddle.load(args.model_path)
    model.set_state_dict(model_state_dict)

    metrics.reset()
    model.eval()
    for _, data in enumerate(test_loader):
        x, y = data
        pred, _, _ = model(x)

        correct = metrics.compute(pred, y)
        metrics.update(correct)
    test_acc = metrics.accumulate()
    print("Test Accuracy: {}".format(test_acc))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    test(args)
