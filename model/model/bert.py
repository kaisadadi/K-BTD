import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from utils.util import calc_accuracy, print_info


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.output_dim = config.getint("model", "output_dim")
        self.batch_size = config.getint('train', 'batch_size')

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.fc = nn.Linear(76800, self.output_dim)

        self.sigmoid = nn.Sigmoid()

        if config.get('train', 'type_of_loss') == 'multi_label_cross_entropy_loss':
            self.multi = True
        else:
            self.multi = False

    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, criterion, config, usegpu, mode="train"):
        x = data['input']
        labels = data['label']
        y, _ = self.bert(x, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        y = self.fc(y)
        if self.multi:
            y = self.sigmoid(y)

        loss = criterion(y, labels, weights = None)
        accu, acc_result = calc_accuracy(y, labels, config)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
