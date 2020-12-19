import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import json
from torch.optim import lr_scheduler
# from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer

from model.loss import get_loss
from utils.util import gen_result, print_info, time_to_str, get_macro_F1, get_micro_F1


def resulting(net, valid_dataset, use_gpu, config):
    net.eval()

    task_loss_type = config.get("train", "type_of_loss")
    criterion = get_loss(task_loss_type)

    running_acc = 0
    running_loss = 0
    cnt = 0
    acc_result = []

    result = []

    while True:
        data = valid_dataset.fetch_data(config)
        # print('fetch data')
        if data is None:
            break
        cnt += 1
        # print(data["label"])
        # gg

        with torch.no_grad():
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if torch.cuda.is_available() and use_gpu:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

        results = net(data, criterion, config, use_gpu, acc_result)

        for a in range(0, len(results["result"])):
            result.append([(cnt - 1) * config.getint("train", "batch_size") + a + 1, results["x"][a].tolist()])

        # print('forward')
        #loss, F1 = results["loss"], results["accuracy_result"]
        outputs, loss, accu = results["x"], results["loss"], results["accuracy"]
        acc_result = results["accuracy_result"]

        running_loss += loss.item()
        running_acc += accu.item()

    # print_info("Valid result:")
    # print_info("Average loss = %.5f" % (running_loss / cnt))
    # print_info("Average accu = %.5f" % (running_acc / cnt))
    # gen_result(acc_result, True)

    net.train()

    return result


def valid_net(net, valid_dataset, use_gpu, config, epoch, writer=None):
    net.eval()

    task_loss_type = config.get("train", "type_of_loss")
    criterion = get_loss(task_loss_type)

    out_result = []

    running_acc = 0
    running_loss = 0
    micro_F1 = 0
    macro_F1 = 0
    cnt = 0
    acc_result = []
    F1_sum = []
    for a in range(int(config.get("model", "output_dim"))):
        F1_sum.append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})

    #doc_list = []
    for idx, data in enumerate(valid_dataset):
        if data is None:
            break
        cnt += 1

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if torch.cuda.is_available() and use_gpu:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = net(data, criterion, config, use_gpu, "eval")
        # print('forward')
        loss, F1 = results["loss"], results["accuracy_result"]
        result = results["result"]
        data["label"] = data["label"].detach().cpu().numpy()
        for k in range(data["label"].shape[0]):
            pre = []
            std = []
            for a in range(len(result[k])):
                if result[k][a] != -1:
                    pre.append(int(result[k][a]))
            for a in range(len(data["label"][k])):
                if data["label"][k][a] != -1:
                    std.append(int(data["label"][k][a]))
            out_result.append({"pre": pre, "std": std})
        #outputs, loss, (level_accu, level_recall, level_F1), F1 = results["result"], results["loss"], results["accuracy"], results["accuracy_result"]
        # loss = results["loss"]
        #acc_result = results["accuracy_result"]

        #doc_list += results['doc_choice'].tolist()

        running_loss += loss.item()
        #running_acc += accu.item()
        for idx, item in enumerate(F1):
            F1_sum[idx]['TP'] += item['TP']
            F1_sum[idx]['FP'] += item['FP']
            F1_sum[idx]['TN'] += item['TN']
            F1_sum[idx]['FN'] += item['FN']



    if writer is None:
        pass
    else:
        writer.add_scalar(config.get("output", "model_name") + " valid loss", running_loss / cnt, epoch)
        writer.add_scalar(config.get("output", "model_name") + " valid accuracy", running_acc / cnt, epoch)


    micro_F1 = get_micro_F1(F1_sum)
    macro_F1 = get_macro_F1(F1_sum)
    # print_info("Valid result:")
    # print_info("Average loss = %.5f" % (running_loss / cnt))
    # print_info("Average accu = %.5f" % (running_acc / cnt))
    # gen_result(acc_result, True)

    #模型eval详细结果写文件
    model_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    #f = open(os.path.join(model_path, "macro_F1.txt"), 'a')
    for idx, item in enumerate(F1_sum):
        precision, recall, f1 = None, None, None
        if item["TP"] == 0:
            if item["FP"] == 0 and item["FN"] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * item["TP"] / (item["TP"] + item["FP"])
            recall = 1.0 * item["TP"] / (item["TP"] + item["FN"])
            f1 = 2 * precision * recall / (precision + recall)
        #f.write("class = {}, TP = {}, FP = {}, TN = {}, FN = {}, F1 = {} \n".format(idx, item["TP"], item["FP"], item["TN"], item["FN"], f1))

    #f.close()

    #json.dump(out_result, open("/home/wke18/Attn.json", "w"))

    net.train()

    #fout = open('/data/disk1/private/xcj/exam/gg.json', 'w')
    #print(json.dumps(doc_list), file = fout)

    return running_loss / cnt, running_acc / cnt, micro_F1, macro_F1

    # print_info("valid end")
    # print_info("------------------------")


def train_net(net, train_dataset, valid_dataset, use_gpu, config):
    epoch = config.getint("train", "epoch")
    learning_rate = config.getfloat("train", "learning_rate")
    task_loss_type = config.get("train", "type_of_loss")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")
    midback = config.getboolean("train", "midback")
    model_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))

    try:
        trained_epoch = config.get("train", "pre_train")
        trained_epoch = int(trained_epoch)
    except Exception as e:
        trained_epoch = 0

    #os.makedirs(os.path.join(config.get("output", "tensorboard_path")), exist_ok=True)

    #if trained_epoch == 0:
    #    shutil.rmtree(
    #        os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    # writer = SummaryWriter(
    #    os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
    #    config.get("output", "model_name"))
    #writer = None

    criterion = get_loss(task_loss_type)

    optimizer_type = config.get("train", "optimizer")
    if optimizer_type == "adam":
        #normal_parameters = list(net.decoder.parameters()) + list(net.term_encoder.parameters()) + list(net.syt_encoder.parameters()) + list(net.encoder_fc.parameters())
        #optimizer = optim.Adam([{"params" :net.bert.parameters(), "lr": 0.01 * learning_rate},
        #                      {"params": net.fc.parameters(), "lr": learning_rate}],weight_decay=config.getfloat("train", "weight_decay"))
        optimizer = optim.Adam(net.parameters(),
                               lr = learning_rate,
                               weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=config.getfloat("train", "momentum"),
                              weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "gamma")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print('** start training here! **')
    print('----------------|----------TRAIN---------------------------|----------VALID--------------------------|----------------|')
    print('  lr    epoch   |   loss           evalu           macro   |   loss           evalu          macro   |      time      | Forward num')
    print('----------------|------------------------------------------|-----------------------------------------|----------------|')
    start = timer()

    for epoch_num in range(trained_epoch, epoch):
        cnt = 0
        total = 0

        train_cnt = 0
        train_loss = 0
        train_acc = 0
        train_micro_F1 = 0
        train_macro_F1 = 0
        F1_0 = 0
        accu_0 = 0
        recall_0 = 0
        F1_1 = 0
        F1_2 = 0

        F1_sum = []
        for a in range(int(config.get("model", "output_dim"))):
            F1_sum.append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})

        exp_lr_scheduler.step(epoch_num)
        lr = 0
        for g in optimizer.param_groups:
            lr = float(g['lr'])
            break

        for idx, data in enumerate(train_dataset):
            #gg
            cnt += 1
            if data is None:
                break

            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if torch.cuda.is_available() and use_gpu:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            optimizer.zero_grad()

            #results = net(data, criterion, config, use_gpu, "train")
            results = net(data, criterion, "train")
            #loss, F1 = results["loss"], results["accuracy_result"]
            outputs, loss, (level_accu, level_recall, level_F1), F1 = results["result"], results["loss"], results["accuracy"], results["accuracy_result"]
            #loss = results["loss"]

            if midback == False:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            #train_acc += level_accu["0"].item()

            train_cnt += 1
            #F1_0 += level_F1["0"]
            #F1_1 += level_F1["1"]
            #accu_0 = level_accu["0"]
            #recall_0 = level_recall["0"]

            for idx, item in enumerate(F1):
                F1_sum[idx]['TP'] += item['TP']
                F1_sum[idx]['FP'] += item['FP']
                F1_sum[idx]['TN'] += item['TN']
                F1_sum[idx]['FN'] += item['FN']

            train_micro_F1 = get_micro_F1(F1_sum)
            train_macro_F1 = get_macro_F1(F1_sum)

            loss = loss.item()
            #accu = accu.item()


            total += config.getint("train", "batch_size")

            if cnt % output_time == 0:
                print('\r', end='', flush=True)
                #print(np.sum(outputs))
                print('%.4f   % 3d    |  %.4f         % .4f        %.4f    |   ????           ?????            ?????|  %s  | %d' % (
                    lr, epoch_num + 1, train_loss / train_cnt, train_micro_F1,  train_macro_F1,
                    time_to_str((timer() - start)), total), end='',
                      flush=True)


        del data

        train_loss /= train_cnt
        train_acc /= train_cnt

        # writer.add_scalar(config.get("output", "model_name") + " train loss", train_loss, epoch_num + 1)
        # writer.add_scalar(config.get("output", "model_name") + " train accuracy", train_acc, epoch_num + 1)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        #torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

        with torch.no_grad():
            valid_loss, valid_accu, valid_micro_F1, valid_macro_F1 = valid_net(net, valid_dataset, use_gpu, config, epoch_num + 1)
        print('\r', end='', flush=True)
        print('%.4f   % 3d    |  %.4f          %.4f         %.4f|  %.4f         % .4f       %.4f|  %s  | %d' % (
            lr, epoch_num + 1, train_loss, train_micro_F1, train_macro_F1, valid_loss, valid_micro_F1, valid_macro_F1,
            time_to_str((timer() - start)), total))


print_info("training is finished!")
