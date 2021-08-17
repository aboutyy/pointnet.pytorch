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
    choices = np.random.choice(len(points), 1024)
    points = points[choices]
    points = torch.tensor(points)
    if torch.cuda.is_available():
        points = points.to('cuda')
    points = points.unsqueeze(dim=0)
    preds = model(points)
    if task == 'segmentation':
        preds = preds.view(-1, num_class)
    else:
        preds = torch.argmax(preds).item()
        print(f'Detected class:{preds}')
        pcd = open3d.geometry.PointCloud()
        points = points.cpu().numpy().squeeze()
        pcd.points = open3d.utility.Vector3dVector(points[:,0:3])
        open3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    check_point = '/home/lijing/PycharmProjects/pointnet.pytorch/checkpoint/modelnet40_normal_resampled2021_08_17-12:31:41_PM_Run(lr=0.001, batch_size=64, shuffle=True, device=\'cuda\', epoch_num=50).pkl'
    point_cloud = '/home/lijing/PycharmProjects/pointnet.pytorch/data/modelnet40_normal_resampled/bathtub/bathtub_0107.txt'
    infer('modelnet', model_checkpoint=check_point, point_cloud_file=point_cloud, task='classification')