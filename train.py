import argparse

import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.metric import Accuracy
from paddle.optimizer import Adam
from paddle.optimizer.lr import StepDecay

from data import ModelNetDataset
from model import PointNet


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
    parser.add_argument("--max_epochs", type=int, default=100, help="max epochs")
    parser.add_argument("--num_workers", type=int, default=32, help="num wrokers")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="/data3/ganyunchong/ModelNet40")

    return parser.parse_args()


def train(args):

    train_data = ModelNetDataset(args.data_dir, "train", num_point=args.num_point)
    test_data = ModelNetDataset(args.data_dir, "test", num_point=args.num_point)

    model = paddle.Model(PointNet())

    model.prepare(
        optimizer=Adam(
            learning_rate=StepDecay(
                learning_rate=args.learning_rate, step_size=20, gamma=0.5
            ),
            parameters=model.parameters(),
        ),
        loss=nn.CrossEntropyLoss(),
        metrics=Accuracy(),
    )

    model.summary(input_size=(args.batch_size, args.num_point, 3), dtype=paddle.float32)

    model.fit(
        train_data=train_data,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        shuffle=True,
        num_workers=args.num_workers,
        verbose=args.verbose,
        log_freq=args.log_freq,
    )

    model.evaluate(
        test_data,
        num_workers=args.num_workers,
        verbose=args.verbose,
        log_freq=args.log_freq,
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
