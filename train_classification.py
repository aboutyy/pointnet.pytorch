import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from utils.run_manager import RunManager, RunBuilder
from pointnet.model import ClassificationPointNet
from collections import OrderedDict
import pointnet.ModelNetDataset as dataset
from datetime import datetime
import os

"""

训练步骤

1.Get batch from the training set
2.Pass batch to the network
3.Calculate the loss
4.Calculate the gradient of the loss function
5.Update the weight using gradient to reduce the loss
6.Repeat steps 1-5 until one epoch is complete
7.Repeat 1-6 for as many epochs required to reach the min loss
"""

# 准备数据
modelnet_train_set = dataset.ModelNetDataset(
    root='data/modelnet40_normal_resampled',
    split='train',
)

# 训练超参数
params = OrderedDict(
    lr=[.001],
    batch_size=[32],
    shuffle=[True],
    device=['cpu'],
    epoch_num=[1]
)

# 多次训练
run_manager = RunManager()
for run in tqdm.tqdm(iterable=RunBuilder.get_runs(params), desc='runs loop'):  # Step 7
    device = torch.device(run.device)
    network = ClassificationPointNet(num_class=10, point_dimension=3).to(device)  # 构建网络
    train_loader = DataLoader(modelnet_train_set, batch_size=run.batch_size, shuffle=run.shuffle)  # 加载数据
    optimizer = optim.Adam(params=network.parameters(), lr=run.lr)  # 设定优化器
    run_manager.begin_run(network, train_loader, run)
    for epoch in tqdm.tqdm(iterable=range(run.epoch_num), desc='epochs loop', leave=False):  # Step 6
        run_manager.begin_epoch()
        for batch in tqdm.tqdm(iterable=train_loader, desc='batches loop', leave=False):   # Step 1
            images = batch[0].to(device)
            labels = batch[1].to(device)
            preds = network(images)  # Step 2

            loss = F.cross_entropy(preds, labels)  # Step 3
            optimizer.zero_grad()
            loss.backward()  # Step 4
            optimizer.step()  # Step 5

            run_manager.track_loss(loss, batch)
            run_manager.track_num_correct(preds, labels)

        run_manager.end_epoch()
    run_manager.end_run()
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    save_model_path = os.path.join('checkpoint', modelnet_train_set.root.split('/')[1] + f'{date}_{run}.pkl')
    torch.save(network.state_dict(), save_model_path)
date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
run_manager.save(f'runs/Results_{date}')
