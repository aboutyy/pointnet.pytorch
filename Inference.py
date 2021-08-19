import torch
import numpy as np
from pointnet.model import ClassificationPointNet, SegmentationPointNet
from pointnet.ModelNetDataset import ModelNetDataset
import open3d

MODELS = {
    'classification': ClassificationPointNet,
    'segmentation': SegmentationPointNet
}

DATASETS = {
    'modelnet':ModelNetDataset
}


def prepare_data(point_file):
    point_cloud = np.loadtxt(point_file,delimiter=',').astype(np.float32)
    # return torch.from_numpy(point_cloud)
    return point_cloud

def infer(dataset, model_checkpoint, point_cloud_file, task):
    """
    Inference, 读取一个训练好的模型，读取一个点云文件，判断其类别，并可视化

    :param dataset: 公开数据集名称
    :param model_checkpoint: 已存好的模型
    :param point_cloud_file: 要推理的点云文件
    :param task: 任务是分了还是分割
    :return:
    """
    if task == 'classification':
        num_class = DATASETS[dataset].get_num_class()
    elif task == 'segmentation':
        num_class = DATASETS[dataset].get_num_class()
    model = MODELS[task](num_class=num_class)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(state_dict=torch.load(model_checkpoint))
    model.eval()
    points = prepare_data(point_cloud_file)
    choices = np.random.choice(len(points), 1000)
    points = points[choices]
    points = torch.tensor(points)
    if torch.cuda.is_available():
        points = points.to('cuda')
    points = points.unsqueeze(dim=0)
    preds, critical_point_indices = model(points)
    if task == 'segmentation':
        preds = preds.view(-1, num_class)
    else:
        preds = torch.argmax(preds).item()
        print(f'Detected class:{preds}')
        pcd = open3d.geometry.PointCloud()
        points = points.cpu().numpy().squeeze()
        pcd.points = open3d.utility.Vector3dVector(points[:,0:3])
        pcd.paint_uniform_color([255,0,0])
        pcd1 = open3d.geometry.PointCloud()
        critical_point_indices = np.asarray(np.unique(critical_point_indices.to('cpu').flatten().numpy()))
        pcd1.points = open3d.utility.Vector3dVector(points[critical_point_indices][:,0:3])
        pcd1.paint_uniform_color([0,255,255])
        open3d.visualization.draw_geometries([pcd, pcd1])

if __name__ == '__main__':
    check_point = '/home/lijing/PycharmProjects/pointnet.pytorch/checkpoint/modelnet40_normal_resampled2021_08_17-12:31:41_PM_Run(lr=0.001, batch_size=64, shuffle=True, device=\'cuda\', epoch_num=50).pkl'
    point_cloud = '/home/lijing/PycharmProjects/pointnet.pytorch/data/modelnet40_normal_resampled/desk/desk_0219.txt'
    infer('modelnet', model_checkpoint=check_point, point_cloud_file=point_cloud, task='classification')