from collections import namedtuple, OrderedDict
from itertools import product
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pandas as pd
import json


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class RunManager:
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None  # RunBuilder 实例中runs的其中一个run，是一个namedtuple 如Run(lr=0.01, batch_size=1000)
        self.run_count = 0
        self.run_data = []  # 是一个nameddict，存储的是最终要记录的变量，例如lr，batch_size，epoch_duration等
        self.run_start_time = None

        self.network = None
        self.tb = None
        self.loader = None

    def begin_run(self, network, loader, run):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        # images, labels = next(iter(self.loader))
        # grid = torchvision.utils.make_grid(images)

        # self.tb.add_image('images', grid)
        # self.tb.add_graph(
        #     self.network
        #     , images.to(getattr(run, 'device', 'cpu'))
        # )

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_num_correct = 0
        self.epoch_loss = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration

        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        # clear_output(wait=True)
        print(df.to_string())

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def save(self, file_name):

        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{file_name}.csv')

        with open(f'{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
