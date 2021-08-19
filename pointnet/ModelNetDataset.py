import torch.utils.data as data
import os
import numpy as np
import tqdm
import pickle


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=1000,
                 split='train',
                 num_category=10,
                 data_augmentation=True,
                 sampling='random'):
        """
        初始化

        :param root: 数据集目录
        :param npoints: 采样点个数
        :param split: 读取数据是train 还是 test
        :param data_augmentation: 是否做数据增强（旋转，平移，重采样），用来构建种类平衡的数据
        """
        assert (split == 'train' or split == 'test')
        self.root = root
        self.npoints = npoints
        self.split = split
        self.data_augmentation = data_augmentation

        self.num_category = num_category

        # 读取文件中的points和labels
        if self.num_category == 10:
            self.cat_file = os.path.join(root, 'modelnet10_shape_names.txt')  # 类别文件位置
        else:
            self.cat_file = os.path.join(root, 'modelnet40_shape_names.txt')

        self.category = [line.strip() for line in open(self.cat_file)]   # 类别列表，如['airplane'，'sofa']
        self.classes = dict(zip(self.category, range(len(self.category))))  # 名字与数字对应的类字典，如 {'airplane'：1, 'sofa':2}
        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.strip() for line in open(os.path.join(root, 'modelnet10_train.txt'))]  # 所有实例的名字
            shape_ids['test'] = [line.strip() for line in open(os.path.join(root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.strip() for line in open(os.path.join(root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.strip() for line in open(os.path.join(root, 'modelnet40_test.txt'))]

        shape_categories = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]   # 对应的实例名字的所属的类
        self.shape_path_with_class = [(shape_categories[i], os.path.join(root, shape_categories[i],shape_ids[split][i]) + '.txt') for i in range(len(shape_ids[split]))]
        self.list_of_samples = []  # 存所有点云数据
        self.list_of_labels = []  # 存所有点云块的标签
        self.data_save_path = os.path.join(root, f'modelnet_{num_category}_{split}_{npoints}_{sampling}.dat')  # 将txt数据转换为numpy的结构化数据，不用每次都处理txt了
        if os.path.exists(self.data_save_path):
            print(f'Reading from {self.data_save_path}...')
            with open(self.data_save_path, 'rb') as f:
                self.list_of_samples, self.list_of_labels = pickle.load(f)
        else:
            for i in tqdm.tqdm(range(len(shape_ids[split])), total=len(shape_ids[split]),desc=f'Processing data from txt(only running in the first time)...'):
                point_set = np.loadtxt(self.shape_path_with_class[i][1], delimiter=',').astype(np.float32)
                choises = np.random.choice(len(point_set), self.npoints, replace=False)  # 随机重采样
                self.list_of_samples.append(point_set[choises])
                self.list_of_labels.append(self.classes[self.shape_path_with_class[i][0]])
            with open(self.data_save_path, 'wb') as f:
                pickle.dump([self.list_of_samples, self.list_of_labels], f)  # 直接从之前存的dat文件中读取点云数据

    def __getitem__(self, index):  # 数据集必须实现的函数，用来获取数据和标签
        return self.list_of_samples[index], self.list_of_labels[index]

    def __len__(self):  # 数据集必须实现的函数
        return len(self.shape_path_with_class)

    @staticmethod
    def get_num_class():
        return 10


if __name__ == '__main__':
    import torch
    train_set = ModelNetDataset('../data/modelnet40_normal_resampled/')
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=12, shuffle=False)
    for point, label in data_loader:
        print(point.shape)
        print(label)