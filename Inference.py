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
    point_cloud = np.loadtxt(point_file)
    return torch.from_numpy(point_cloud)

def infer(dataset, model_checkpoint, point_cloud_file, task):
    if task == 'classification':
        num_classes = DATASETS[dataset].num_category
    elif task == 'segmentation':
        num_classes = DATASETS[dataset].num_category
    model = MODELS[task](num_classes=num_classes)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(state_dict=torch.load(model_checkpoint))
    model.eval()
    points = prepare_data('xx.txt')
    points = torch.tensor(points)
    if torch.cuda.is_available():
        points.cuda()
    points = points.unsqueeze(dim=0)
    preds = model(points)
    if task == 'segmentation':
        preds = preds.view(-1, num_classes)
    else:
        preds = torch.argmax(preds).item()
        print(f'Detected class:{preds}')
        open3d.visualization.draw([points.squeeze()])

