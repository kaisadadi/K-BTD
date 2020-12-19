import torch

from utils.util import check_multi


def top1(outputs, label, config, result=None):
    if check_multi(config):
        if len(label[0]) != len(outputs[0]):
            raise ValueError('Input dimensions of labels and outputs must match.')

        outputs = outputs.data
        labels = label.data

        if result is None:
            result = []

        total = 0
        nr_classes = outputs.size(1)
        while len(result) < nr_classes:
            result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        for i in range(nr_classes):
            outputs1 = (outputs[:, i] >= 0.5).long()
            labels1 = (labels[:, i].float() >= 0.5).long()
            total += int((labels1 * outputs1).sum())
            total += int(((1 - labels1) * (1 - outputs1)).sum())

            if result is None:
                continue

            # if len(result) < i:
            #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

            result[i]["TP"] += int((labels1 * outputs1).sum())
            result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
            result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
            result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

        return torch.Tensor([1.0 * total / len(outputs) / len(outputs[0])]), result
    else:

        if not (result is None):
            # print(label)
            id1 = torch.max(outputs, dim=1)[1]
            # id2 = torch.max(label, dim=1)[1]
            id2 = label
            nr_classes = outputs.size(1)
            while len(result) < nr_classes:
                result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
            for a in range(0, len(id1)):
                # if len(result) < a:
                #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

                it_is = int(id1[a])
                should_be = int(id2[a])
                if it_is == should_be:
                    result[it_is]["TP"] += 1
                else:
                    result[it_is]["FP"] += 1
                    result[should_be]["FN"] += 1
        pre, prediction = torch.max(outputs, 1)
        prediction = prediction.view(-1)

        return torch.mean(torch.eq(prediction, label).float()), result


def top2(outputs, label, config, result=None):
    if not (result is None):
        # print(label)
        id1 = torch.max(outputs, dim=1)[1]
        # id2 = torch.max(label, dim=1)[1]
        id2 = label
        nr_classes = outputs.size(1)
        while len(result) < nr_classes:
            result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
        for a in range(0, len(id1)):
            # if len(result) < a:
            #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

            it_is = int(id1[a])
            should_be = int(id2[a])
            if it_is == should_be:
                result[it_is]["TP"] += 1
            else:
                result[it_is]["FP"] += 1
                result[should_be]["FN"] += 1

    _, prediction = torch.topk(outputs, 2, 1, largest=True)
    prediction1 = prediction[:, 0:1]
    prediction2 = prediction[:, 1:]

    prediction1 = prediction1.view(-1)
    prediction2 = prediction2.view(-1)

    return torch.mean(torch.eq(prediction1, label).float()) + torch.mean(torch.eq(prediction2, label).float()), result

def level_accuracy(statis_pre, statis_std):
    #计算每层的accuracy, recall和F1，直接采用每个平均的方式
    accuracy, recall, F1 = {}, {}, {}
    for key in statis_pre.keys():
        if key == "final":
            continue
        TP, FP, TN, FN = 0, 0, 0, 0
        for idx1, item in enumerate(statis_pre[key]):
            for idx2 in range(len(item)):
                if statis_std[key][idx1][idx2] == 1:
                    if item[idx2] == 1:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if item[idx2] == 1:
                        FP += 1
                    else:
                        TN += 1
        if TP == 0:
            accuracy[key] = 0
            recall[key] = 0
            F1[key] = 0
        else:
            accuracy[key] = TP / (TP + FP)
            recall[key] = TP / (TP + FN)
            F1[key] = 2 * accuracy[key] * recall[key] / (accuracy[key] + recall[key])
    #对final进行特殊处理
    nr_classes = 100 #需要根据output_dim修改
    result = []
    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    for idx, item in enumerate(statis_std["final"]):
        pre = [0] * nr_classes
        std = [0] * nr_classes
        for disease in item:
            #print(disease)
            if disease != -1:
                std[disease] = 1
        for disease in statis_pre["final"][idx]:
            pre[disease] = 1
        for a in range(nr_classes):
            if std[a] == 1:
                if pre[a] == 1:
                    result[a]["TP"] += 1
                else:
                    result[a]["FN"] += 1
            else:
                if pre[a] == 1:
                    result[a]["FP"] += 1
                else:
                    result[a]["TN"] += 1

    #前一项是分层的结果，后一项是最终的结果，保持与result一致
    return (accuracy, recall, F1), result

