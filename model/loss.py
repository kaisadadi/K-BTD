import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def get_loss(task_loss_type):
    if task_loss_type == "cross_entropy_loss":
        criterion = cross_entropy_loss
    elif task_loss_type == "nll_loss":
        criterion = nn.NLLLoss()
    elif task_loss_type == "focal_loss":
        criterion = FocalLoss()
    elif task_loss_type == "multi_label_cross_entropy_loss":
        criterion = multi_label_cross_entropy_loss
    elif task_loss_type == "EM_and_cross_entropy_loss":
        criterion = EM_and_cross_entropy_loss
    elif task_loss_type == "multi_label_cross_entropy_loss_with_weight":
        criterion = multi_label_cross_entropy_loss_with_weight
    else:
        raise NotImplementedError

    return criterion


def EM_and_cross_entropy_loss(option_prob, option_output, labels):
    loss_cross = cross_entropy_loss(option_output, labels)
    label_one_shot = torch.zeros(option_prob.size()).cuda()
    label_one_shot.scatter_(dim = 1, index = labels.unsqueeze(1), value = 1)

    option = label_one_shot.mul(option_prob) + 1 - (1 - label_one_shot).mul(option_prob)
    option_prob_log = - torch.log(option)
    loss = torch.mean(option_prob_log)
    
    return loss_cross #+ loss


def multi_label_cross_entropy_loss(outputs, labels, weights):
    outputs = outputs.view(-1, 1)
    labels = labels.view(-1, 1)
    outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
    labels = labels.float()
    res = - labels * torch.log(outputs) - (1 - labels) * torch.log(1 - outputs)
    res = torch.sum(torch.sum(res, dim = 1))
    return res

def multi_label_cross_entropy_loss_with_weight(outputs, labels, weights):
    weights = np.array(weights)
    mean = np.mean(weights)
    ratio = torch.from_numpy(mean / weights).float().cuda()
    outputs = outputs.view(-1, 1)
    labels = labels.view(-1, 1)
    outputs = torch.clamp(outputs, 1e-6, 1 - 1e-6)
    labels = labels.float()
    res = - labels * torch.log(outputs) - (1 - labels) * torch.log(1 - outputs)
    res = res * ratio
    res = torch.sum(torch.sum(res, dim=1))
    return res


def cross_entropy_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
