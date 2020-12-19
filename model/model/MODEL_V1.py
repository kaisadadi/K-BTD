import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import numpy as np
import json
import jieba
import pickle
from tree.tree_gen import Node


class Bert_encoder(nn.Module):
    def __init__(self, config):
        super(Bert_encoder, self).__init__()
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

    def forward(self, data):
        y, _ = self.bert(data, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        return y


class raw_decoder(nn.Module):
    def __init__(self, config):
        super(raw_decoder, self).__init__()
        self.output_dim = int(config.get("model", "output_dim"))
        self.hidden_size = int(config.get("model", "hidden_size"))
        self.input_dim = config.getint("model", "term_hidden_size") + config.getint('model', 'filters') * len(config.get('model', 'n_grams').split(","))
        self.LSTM = nn.LSTM(input_size = self.input_dim,
                            hidden_size = self.hidden_size,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = False)

        self.fc = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, decoder_input, h, c):
        x, (h_out, c_out) = self.LSTM(decoder_input.unsqueeze(1), (h, c))
        x = self.fc(x.squeeze())
        x = self.sigmoid(x)
        return x, h_out, c_out


class MODEL_V1(nn.Module):
    def __init__(self, config):
        super(MODEL_V1, self).__init__()
        self.encoder = Bert_encoder(config)
        self.decoder = raw_decoder(config)
        self.hidden_size = int(config.get("model", "hidden_size"))
        #these data files, you need to generate
        #the pre-generated external knowledge
        self.data = json.load(open("__YOUR_PATH__/syp_dict.json", "r"))
        #embedding path, published by Tencent
        self.embedding = json.load(open(config.get("data", "Embedding_Path"), "rb"))
        self.max_len = int(config.get("model", "syp_max_len"))
        self.term_max_len = int(config.get("model", "term_max_len"))
        #the pre-generated ICD tree structure
        self.tree = pickle.load(open("__YOUR_PATH__/tree.pkl", "rb"))
        #threthold for judge net
        self.threthold = config.getfloat("model", "threthold")
        self.syt_encoder = TextEncoder(config)
        self.term_encoder = TermEncoder(config)
        self.encoder_fc = nn.Linear(76800, self.hidden_size)
        self.init_jieba()

    def init_jieba(self):
        jieba.initialize()
        #use jieba to separate words
        for key in self.data.keys():
            self.data[key]['name'] = list(jieba.cut(self.data[key]['name']))
            self.data[key]['syptom'] = list(jieba.cut(self.data[key]['syptom']))
                    

    def forward(self, data, criterion, use_gpu, acc_result=None):
        encoded = self.encoder.forward(data['input'])
        encoded_fc = self.encoder_fc(encoded)
        label = data['label']
        h_0 = torch.zeros(1, 1, self.hidden_size).cuda()
        root = 0
        loss = 0
        for batch in range(data['input'].shape[0]):
            loss += self.Node_loss(root, encoded_fc[batch], h_0, 0, label[batch], criterion)
        return {"loss":loss / data['label'].shape[0]}


    def Node_loss(self, node, last_h, last_c, depth, target, criterion):
        #node: current position
        #last_h, last_c: flowing vector
        #depth: current depth in the ICD tree, root depth = 0
        #target: ground-truth label
        loss = 0
        #target_new: child node label
        target_new = [0] * len(self.tree[node].children)
        bs = len(self.tree[node].children)
        for target_node in target:
            while target_node != 0:
                for idx, son in enumerate(self.tree[node].children):
                    if target_node == son:
                        target_new[idx] = 1
                target_node = self.tree[target_node].father
        #symptom_doc: external knowledge
        #term_doc: node name
        symptom_doc = []
        term_doc = []
        for son in self.tree[node].children:
            symptom_doc.append(self.data[self.tree[son].ID]['syptom'])
            term_doc.append(self.data[self.tree[son].ID]['name'])
        #embedding
        syp2vec = []
        for idx_doc, doc in enumerate(symptom_doc):
            syp2vec.append([])
            for idx_word, word in enumerate(doc):
                if word in self.embedding.keys():
                    syp2vec[-1].append(self.embedding[word])
                else:
                    syp2vec[-1].append([0.00] * 200)  
            #padding
            if len(syp2vec[-1]) < self.max_len:
                while len(syp2vec[-1]) < self.max_len:
                    syp2vec[-1].append([0.00] * 200)
            else:
                syp2vec[-1] = syp2vec[-1][:self.max_len]
        term2vec = []
        for idx_doc, doc in enumerate(term_doc):
            term2vec.append([])
            for idx_word, word in enumerate(doc):
                if word in self.embedding.keys():
                    term2vec[-1].append(self.embedding[word])
                else:
                    term2vec[-1].append([0.00] * 200)
            if len(term2vec[-1]) < self.term_max_len:
                while len(term2vec[-1]) < self.term_max_len:
                    term2vec[-1].append([0.00] * 200)
            else:
                term2vec[-1] = term2vec[-1][:self.term_max_len]
        #Judge by a LSTM 
        term_doc = torch.from_numpy(np.array(term2vec, np.float32)).cuda()
        symptom_doc = torch.from_numpy(np.array(syp2vec, np.float32)).cuda()
        encoded_term = self.term_encoder.forward(term_doc)
        encoded_symptom = self.syt_encoder.forward(symptom_doc)
        decoder_output, decoder_h, decoder_c = self.decoder.forward(torch.cat([encoded_symptom, encoded_term], dim=1), last_h.repeat(1, bs, 1), last_c.repeat(1, bs, 1))

        loss += np.exp(-depth) * criterion(decoder_output, torch.from_numpy(np.array(target_new)).cuda().long())
        decoder_output = decoder_output.detach().cpu().numpy()
        for idx, out in enumerate(decoder_output):
            if out > self.threthold and len(self.tree[self.tree[node].children[idx]].children) > 0 and target_new[idx] == 1:
                loss += self.Node_loss(self.tree[node].children[idx], decoder_h[:, idx, :], decoder_c[:, idx, :], depth + 1, target, criterion)
        return loss


class TermEncoder(nn.Module):
    def __init__(self, config):
        super(TermEncoder, self).__init__()
        self.hidden_size = int(config.get("model", "term_hidden_size"))
        self.GRU = nn.GRU(input_size = int(config.get("data", "vec_size")),
                          hidden_size = self.hidden_size,
                          num_layers = 1,
                          batch_first = True)

    def forward(self, x):
        x, _ = self.GRU(x)
        return x[:, -1, :].squeeze(1)


class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.n_grams = config.getint("model", "n_grams").split(",")   

        self.convs = []
        for gram_size in self.n_grams:
            self.convs.append(nn.Conv2d(1, config.getint('model', 'filters'), (int(gram_size), self.data_size)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')
        self.relu = nn.ReLU()

    def init_multi_gpu(self, device):
        for conv in self.convs:
            conv = nn.DataParallel(conv)
        self.fc = nn.DataParallel(self.fc)
        self.relu = nn.DataParallel(self.relu)
        self.sigmoid = nn.DataParallel(self.sigmoid)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1, self.data_size)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(x.shape[0], -1)
            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        return conv_out



