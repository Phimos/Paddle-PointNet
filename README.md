# Paddle-PointNet
*Use PaddlePaddle to implementate PointNet (Classifier Only)*

**Paper:** [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)

**Competition Page:** [PaddlePaddle AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/106)

**PointNet Architecture:**
![arch](arch.png)

**Other Version Implementation:**
* [TensorFlow (Official)](https://github.com/charlesq34/pointnet)
* [PyTorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)


## Metric
* Classification Accuracy 89.2 on ModelNet40 Dataset


## Usage

### Data Preparation
Download [alignment ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `modelnet40_normal_resampled/`. The same dataset as the PyTorch version implementation.

```
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
unzip modelnet40_normal_resampled.zip
```

### Train
```
python train.py
```

The model will be saved as `pointnet.pdparams` by default.

### Test
```
python test.py
```

## Classification result on ModelNet40
|Model|Accuracy|
|-|-|
|PointNet (Official)|89.2|
|PointNet (PyTorch)|90.6|
|PointNet (PaddlePaddle)|89.4|