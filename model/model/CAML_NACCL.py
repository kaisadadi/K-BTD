import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import calc_accuracy, print_info


class CAML(nn.Module):
    def __init__(self, config):
        super(CAML, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.output_dim = config.getint("model", "output_dim")
        self.batch_size = config.getint('train', 'batch_size')

        self.n_grams = config.getint("model", "n_grams").split(",")   
        self.filter_num = config.getint('model', 'filters')
        self.convs = []
        for gram_size in self.n_grams:
            self.convs.append(nn.Conv2d(in_channels = 1, 
                                        out_channels = self.filter_num,
                                        kernel_size = (int(gram_size), self.data_size),
                                        stride = 1,
                                        padding = (int(gram_size) / 2, 0),
                                        dilation = 1
                                        ))
        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')

        self.fc1 = nn.Linear(self.feature_len, self.output_dim)
        self.fc2 = nn.Linear(self.feature_len, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, data, criterion, config, usegpu, acc_result=None):
        x = data['input']
        labels = data['label']

        x = x.view(x.shape[0], 1, -1, self.data_size)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1).squeeze(3).permute(0, 2, 1)

        #attention
        attn = nn.Softmax(dim = 1)(self.fc1(conv_out)).permute(0, 2, 1)
        attn_out = self.fc2(torch.bmm(attn, conv_out)).squeeze(2)

        y = self.sigmoid(attn_out)

        loss = criterion(y, labels, weights = None)
        accu, acc_result = calc_accuracy(y, labels, config, None)
        return {"loss": loss, "accuracy": accu, "result": torch.ge(y, 0.5).cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
