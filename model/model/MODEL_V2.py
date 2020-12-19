import torch
import torch.nn as nn
import json
import pickle
import jieba
import numpy as np
from utils.accuracy import level_accuracy


class JudgeNet(nn.Module):
    def __init__(self, config):
        super(JudgeNet, self).__init__()
        self.hidden_size = config.getint("model", "hidden_size")
        self.max_len = config.getint("model", "max_len")
        self.syp_max_len = config.getint("model", "syp_max_len")
        self.lambd = config.getfloat("model", "lambda")
        
        self.semantic_fc = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.ReLu(),
            nn.Linear(self.hidden_size, 1)
        )         
        self.word_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.GRU = nn.GRU(input_size = self.syp_max_len,  
                          hidden_size = self.hidden_size,
                          batch_first = True,
        )
        self.word_fc2 = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.row_softmax = nn.Softmax(dim = 3)
        self.col_softmax = nn.Softmax(dim = 2)
    
    def semantic_level_score(self, Fn, Am):
        concat_input = torch.cat([Fn, Am, Fn * Am, torch.abs(Fn - Am)], dim = 1)
        s1 = self.sigmoid(self.semantic_fc(concat_input))
        return s1

    def word_level_score(self, T, Cm):
        T = self.word_fc1(T)
        S = torch.tanh(torch.bmm(T, Cm.permute(0, 2, 1)))
        row_S = self.row_softmax(S)
        col_S = self.col_softmax(S)  

        xi = torch.sum(col_S, dim=2, keepdim=False)  #importance measurement

        P = torch.bmm(row_S, Cm)
        P_hat, _ = self.GRU(P)

        output, _ = torch.max(torch.cat([P_hat, T], dim = 2), dim=1, keepdim=False)
        s2 = self.sigmoid(self.word_fc2(output))
        return s2, xi

    def forward(self, T, Cm, Fn, Am):
        s1 = self.semantic_level_score(Fn, Am)
        s2, xi = self.word_level_score(T, Cm)
        return self.lambd * s1 + (1 - self.lambd) * s2, xi


class MODEL_V2(nn.Module):
    def __init__(self, config):
        super(MODEL_V2, self).__init__()
        self.encoder_LSTM = nn.LSTM(input_size = 200,
                                    hidden_size = config.getint("model", "hidden_size"),
                                    num_layers = 1,
                                    batch_first = True,
                                    dropout = 0,
                                    bidirectional = True)

        self.symp_encoder = TextEncoder(config)

        self.judge_net = JudgeNet(config)
        self.max_len = int(config.get("model", "syp_max_len"))
        self.teacher_forcing = config.getboolean("train", "teacher_forcing")
        self.midback = config.getboolean("train", "midback")
        self.threthold = config.getfloat("model", "threthold")

        self.data = json.load(open("__YOUR_PATH__/syp_dict.json", "r"))
        self.embedding = json.load(open(config.get("data", "Embedding_Path"), "rb"))
        self.tree = pickle.load(open("__YOUR_PATH__/sub_tree.pkl", "rb"))
        self.disease_num2ICD = json.load(open("__YOUR_PATH__/disease_num2ICD.json", "r"))
        self.ICD2pos = pickle.load(open("__YOUR_PATH__n/sub_id2pos.pkl", "rb"))
        self.sub_tree_statis = pickle.load(open("__YOUR_PATH__/sub_tree_statis.pkl", "rb"))
        self.ICD2disease_num = {}
        for key in self.disease_num2ICD.keys():
            self.ICD2disease_num[self.disease_num2ICD[key]] = int(key)

        #init
        self.init_jieba()
        self.init_weight()

    def init_jieba(self):
        jieba.initialize()
        for key in self.data.keys():
            self.data[key]['name'] = list(jieba.cut(self.data[key]['name']))
            self.data[key]['syptom'] = list(jieba.cut(self.data[key]['syptom']))

    def init_weight(self):
        nn.init.xavier_normal(self.encoder_LSTM.all_weights[0][0])
        nn.init.xavier_normal(self.encoder_LSTM.all_weights[0][1])
        nn.init.xavier_normal(self.encoder_LSTM.all_weights[1][0])
        nn.init.xavier_normal(self.encoder_LSTM.all_weights[1][1])

        nn.init.xavier_normal(self.symp_LSTM.all_weights[0][0])
        nn.init.xavier_normal(self.symp_LSTM.all_weights[0][1])
        nn.init.xavier_normal(self.symp_LSTM.all_weights[1][0])
        nn.init.xavier_normal(self.symp_LSTM.all_weights[1][1])


    def forward(self, data, criterion, use_gpu, acc_result=None, mode = "train", optimizer = None):
        encoded_time, (Fn, _) = self.encoder_LSTM(data['input'])
        label = data['label']
        root = 0
        loss = 0
        self.statis_pre = {"0": [], "1": [], "2": [], "3": [], "final":[]}  
        self.statis_std = {"0": [], "1": [], "2": [], "3": [], "final":label.detach().cpu()}  
        for batch in range(data['input'].shape[0]):
            self.statis_pre["final"].append([])
            loss += self.Node_loss(root, encoded_time[batch], Fn[batch], 0, label[batch], criterion, mode, optimizer)

        (accu, recall, F1), accu_result = level_accuracy(self.statis_pre, self.statis_std)
        return {"loss": loss / data['label'].shape[0], "result": self.statis_pre["final"], "accuracy": (accu, recall, F1), "accuracy_result": accu_result}


    def Node_loss(self, node, T, Fn, depth, target, criterion, mode, optimizer = None):
        #node: current position
        #T: clinical notes feature
        #Fn: flowing vector
        #depth: current depth in the ICD tree, root depth = 0
        #target: ground-truth label
        loss = 0
        target_new = [0] * len(self.tree[node].children)
        target_transferred = []
        for disease in target:
            if disease == -1:
                break
            target_transferred.append(self.ICD2pos[self.disease_num2ICD[str(disease.detach().cpu().numpy())]])
        bs = len(self.tree[node].children)
        for target_node in target_transferred:
            while target_node != 0:
                for idx, son in enumerate(self.tree[node].children):
                    if target_node == son:
                        target_new[idx] = 1
                target_node = self.tree[target_node].father
        symptom_doc = []
        term_doc = []
        for son in self.tree[node].children:
            symptom_doc.append(self.data[self.tree[son].ID]['syptom'])
            term_doc.append(self.data[self.tree[son].ID]['name'])
        # embedding
        syp2vec = []
        for idx_doc, doc in enumerate(symptom_doc):
            syp2vec.append([])
            for idx_word, word in enumerate(doc):
                if word in self.embedding.keys():
                    syp2vec[-1].append(self.embedding[word])
                else:
                    syp2vec[-1].append([0.00] * 200)  
            # padding
            if len(syp2vec[-1]) < self.max_len:
                while len(syp2vec[-1]) < self.max_len:
                    syp2vec[-1].append([0.00] * 200)
            else:
                syp2vec[-1] = syp2vec[-1][:self.max_len]
        symptom_doc = torch.from_numpy(np.array(syp2vec, np.float32)).cuda()
        encoded_symptom, summed_output = self.symp_encoder.forward(symptom_doc)
        prediction, xi = self.judge_net.forward(T.unsqueeze(0).repeat(bs, 1, 1), encoded_symptom, Fn.repeat(bs, 1), summed_output)

        loss += criterion(prediction, torch.from_numpy(np.array(target_new)).cuda().long(), self.sub_tree_statis[node].ratio)
        if self.midback == True:
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
        # expands node
        decoder_output = prediction.detach().cpu().numpy()
        decoder_output = 1.0 * (decoder_output > self.threthold)  
        self.statis_pre[str(depth)].append(list(decoder_output))
        self.statis_std[str(depth)].append(target_new)
        if mode == "eval":
            for idx, out in enumerate(decoder_output):
                if out == 1 and len(self.tree[self.tree[node].children[idx]].children) == 0:
                    self.statis_pre["final"][-1].append(self.ICD2disease_num[self.tree[self.tree[node].children[idx]].ID])
                if out == 1 and len(self.tree[self.tree[node].children[idx]].children) > 0:
                    loss += self.Node_loss(self.tree[node].children[idx],
                                                    T,
                                                    Fn,
                                                    depth + 1,
                                                    target,
                                                    criterion, mode)
        elif self.teacher_forcing == False:
            for idx, out in enumerate(decoder_output):
                if out == 1 and len(self.tree[self.tree[node].children[idx]].children) == 0 and target_new[idx] == 1:
                    self.statis_pre["final"][-1].append(self.ICD2disease_num[self.tree[self.tree[node].children[idx]].ID])
                if out == 1 and len(self.tree[self.tree[node].children[idx]].children) > 0 and target_new[idx] == 1:
                    loss += self.Node_loss(self.tree[node].children[idx],
                                                    T,
                                                    Fn,
                                                    depth + 1,
                                                    target,
                                                    criterion, mode)
        else:
            for idx, out in enumerate(decoder_output):
                if out == 1 and len(self.tree[self.tree[node].children[idx]].children) == 0 and target_new[idx] == 1:
                    self.statis_pre["final"][-1].append(self.ICD2disease_num[self.tree[self.tree[node].children[idx]].ID])
                if len(self.tree[self.tree[node].children[idx]].children) > 0 and target_new[idx] == 1:
                    loss += self.Node_loss(self.tree[node].children[idx],
                                                    T,
                                                    Fn,
                                                    depth + 1,
                                                    target,
                                                    criterion, mode)
        return loss



class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.data_size = config.getint("data", "vec_size")
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
        self.relu = nn.ReLU()
        self.attn_fc = nn.Sequential(
                        nn.Linear(self.filter_num * len(self.n_grams), int(self.filter_num / 2)),
                        nn.ReLu(),
                        nn.Linear(int(self.filter_num / 2), 1)
        )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1, self.data_size)
        conv_out = []
        for conv in self.convs:
            y = self.relu(conv(x))   
            y = y.permute(0, 2, 1)
            conv_out.append(y)
        conv_out = torch.cat(conv_out, dim=2)
        attn_weight = self.attn_fc(conv_out).repeat(1, 1, conv_out.shape[2])
        summed_output = torch.sum(conv_out * attn_weight, dim = 1, keepdim = False)  
        return y, summed_output