import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# 构建模型
class BasePointNet(nn.Module):
    def __init__(self, point_dimension=3):
        super(BasePointNet, self).__init__()
        self.point_dimension = point_dimension
        #  layer1: 3*n to 64*n
        self.conv1 = nn.Conv1d(point_dimension,64,1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        #  layer2: 64*n to 64*n
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.batch_norm2 = nn.BatchNorm1d(64)
        #  layer3: 64*n to 64*n
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.batch_norm3 = nn.BatchNorm1d(64)
        #  layer4:64*n to 128*n
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.batch_norm4 = nn.BatchNorm1d(128)
        #  layer5:128*n to 1024*n
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.batch_norm5 = nn.BatchNorm1d(1024)

    def forward(self, t):
        t = t[:, :, 0:self.point_dimension]
        t = t.transpose(2, 1)  # 将n*3的矩阵转为3*n，t的shape为（batch_size，number_of_points，number_of_data_dimensions）
        t = F.relu(self.batch_norm1(self.conv1(t)))
        t = F.relu(self.batch_norm2(self.conv2(t)))
        t = F.relu(self.batch_norm3(self.conv3(t)))

        t = F.relu(self.batch_norm4(self.conv4(t)))
        t = F.relu(self.batch_norm5(self.conv5(t)))

        max_pool = nn.MaxPool1d(t.shape[2], return_indices=True)  # maxpooling
        t, critial_indices = max_pool(t)  ## 获得critical points的index

        t = t.view(-1, 1024)  # 展平1024*1 的矩阵为一个向量，作为后面全连接层的输入
        critial_indices = critial_indices.view(-1, 1024)
        return t, critial_indices


class ClassificationPointNet(nn.Module):
    def __init__(self, num_class, point_dimension=3):
        super(ClassificationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.batch_nomr1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.batch_nomr2 = nn.BatchNorm1d(256)

        self.out = nn.Linear(in_features=256, out_features=num_class)

    def forward(self, t):
        t, critial_indices = self.base_pointnet(t)
        t = F.relu(self.batch_nomr1(self.fc1(t)))
        t = F.relu(self.batch_nomr2(self.fc2(t)))
        # TODO  add dropout here
        t = self.out(t)

        # return F.log_softmax(t, dim=1)  # 如果loss函数用的cross_entropy则不用计算softmax值
        return t, critial_indices


class SegmentationPointNet(nn.Module):
    def __init__(self, num_class, point_dimension=3):
        super(SegmentationPointNet, self).__init__()

    def forward(self, t):
        return t


if __name__ == '__main__':
    network = ClassificationPointNet(10, 3)
    data_set = torch.rand(3, 1024, 3)
    predictions = network(data_set)
    print(predictions)