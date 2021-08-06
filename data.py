import glob
import os

import numpy as np
import paddle
import trimesh
from paddle.io import Dataset


def normalize_point_cloud(data: np.ndarray):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std.max()
    return data


def jitter_point_cloud(data: np.ndarray, sigma: float = 0.02, clip: float = 0.05):
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(*data.shape), -clip, clip)
    data = data + jittered_data
    return data


def random_rotate_point_cloud(data: np.ndarray):
    angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array(
        [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]], dtype=data.dtype
    )
    data = data @ rotation_matrix
    return data


class ModelNetDataset(Dataset):
    def __init__(self, data_dir, split: str = "train", num_point: int = 2048):
        super().__init__()
        assert split in ["train", "test"]
        self.split = split
        self.num_point = num_point

        classes = os.listdir(data_dir)
        classes = sorted(classes)
        self.class2idx = {name: idx for idx, name in enumerate(classes)}

        self.filepaths = glob.glob(os.path.join(data_dir, "**", split, "*.off"))

    def _label(self, filepath: str):
        class_name = filepath.split("/")[-3]
        return self.class2idx[class_name]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        data = trimesh.load(filepath).sample(self.num_point).astype(np.float32)
        data = normalize_point_cloud(data)

        if self.split == "train":
            data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        label = self._label(filepath)
        return paddle.to_tensor(data), label


if __name__ == "__main__":
    dataset = ModelNetDataset()
    dataset[0]
