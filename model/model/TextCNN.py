import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import calc_accuracy, print_info


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.output_dim = config.getint("model", "output_dim")
        self.batch_size = config.getint('train', 'batch_size')

        self.n_grams = config.getint("model", "n_grams").split(",")   
        self.convs = []
        for gram_size in self.n_grams:
            self.convs.append(nn.Conv2d(1, config.getint('model', 'filters'), (int(gram_size), self.data_size)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')
        self.fc = nn.Linear(self.feature_len, self.output_dim)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        if config.get('train', 'type_of_loss') == 'multi_label_cross_entropy_loss':
            self.multi = True
        else:
            self.multi = False

    def init_multi_gpu(self, device):
        for conv in self.convs:
            conv = nn.DataParallel(conv)
        self.fc = nn.DataParallel(self.fc)
        self.relu = nn.DataParallel(self.relu)
        self.sigmoid = nn.DataParallel(self.sigmoid)

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        x = data['input']
        labels = data['label']

        x = x.view(x.shape[0], 1, -1, self.data_size)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(x.shape[0], -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        y = self.fc(conv_out)
        if self.multi:
            y = self.sigmoid(y)

        loss = criterion(y, labels, weights = None)
        accu, acc_result = calc_accuracy(y, labels, config, None)
        return {"loss": loss, "accuracy": accu, "result": torch.ge(y, 0.5).cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
