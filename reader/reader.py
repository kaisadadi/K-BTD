import torch
import os
import torch.utils.data as data
import pickle
import json
import numpy as np
import jieba
from pytorch_pretrained_bert import BertTokenizer
import os



class EHR_Dataset(data.Dataset):
    def __init__(self, config, mode):
        self.use_bert = config.getboolean("model", "use_bert")
        f = open(os.path.join(config.get("data", mode + "_data_path")), "rb")
        self.data = pickle.load(f)
        f.close()
        f = open(os.path.join(config.get("data", "Embedding_Path")), "rb")
        self.embedding = json.load(f)
        f.close()
        self.zero_vec = []
        self.out_vec = int(config.get("model", "output_dim"))
        self.max_len = int(config.get("model", "max_len"))
        self.model = config.get("model", "name")
        self.disease_num2ICD = json.load(open("__YOUR_PATH__/disease_num2ICD.json", "r"))
        self.ICD2pos = pickle.load(open("__YOUR_PATH__/sub_id2pos.pkl", "rb"))
        for a in range(200):
            self.zero_vec.append(0.0)
        if self.use_bert == False:
            jieba.initialize()
        else:
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config.get("model", "bert_path"),
                                                           "vocab.txt"))


    def __getitem__(self, index):
        if self.use_bert == True:
            record = self.data[index]
            for key in self.data[index].keys():
                if key == "急诊诊断":
                    continue
                text = record[key]
                word2vec = []
                for word in text:
                    if word in self.tokenizer.vocab.keys():
                        word2vec.append(self.tokenizer.vocab[word])
                    else:
                        word2vec.append(self.tokenizer.vocab["[UNK]"])
                if len(word2vec) > self.max_len:
                    word2vec = word2vec[:self.max_len]
                else:
                    while len(word2vec) < self.max_len:
                        word2vec.append(self.tokenizer.vocab["[PAD]"])
                record[key] = word2vec
            if "MODEL" in self.model:
                jzzd = np.zeros(32, np.int32)
                for idx, disease in enumerate(record["急诊诊断"]):
                    disease = int(disease)
                    jzzd[idx] = disease
                record["急诊诊断"] = jzzd
            else:
                jzzd = np.zeros([self.out_vec], np.float32)
                for disease in record["急诊诊断"]:
                    disease = int(disease)
                    jzzd[disease] = 1
                record["急诊诊断"] = jzzd
            return record
        record = self.data[index]
        for key in self.data[index].keys():
            if key == "急诊诊断":
                continue
            text = list(jieba.cut(record[key]))
            word2vec = []
            for word in text:
                if word in self.embedding.keys():
                    word2vec.append(self.embedding[word])
                else:
                    word2vec.append(self.zero_vec)
            if len(word2vec) > self.max_len:
                word2vec = word2vec[:self.max_len]
            else:
                while len(word2vec) < self.max_len:
                    word2vec.append(self.zero_vec)
            record[key] = word2vec

        if "MODEL" in self.model:
            jzzd = np.zeros(32, np.int32) - 1
            for idx, disease in enumerate(record["急诊诊断"]):
                disease = int(disease)
                jzzd[idx] = disease
        else:
            jzzd = np.zeros([self.out_vec], np.float32)
            for disease in record["急诊诊断"]:
                disease = int(disease)
                jzzd[disease] = 1
        record["急诊诊断"] = jzzd
        return record

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    diagnosis = []
    xbs = []
    zs = []
    fzjc = []
    tgjc = []
    out_data = {"现病史": xbs, "主诉": zs, "辅助检查": fzjc, "体格检查": tgjc, "急诊诊断": diagnosis}
    for idx, record in enumerate(data):
        for key in record.keys():
            if key == "急诊诊断":
                out_data[key].append(torch.from_numpy(record[key]))
                continue
            out_data[key].append(torch.from_numpy(np.array(record[key])))
    real_out_data = {"input": torch.stack(xbs, dim=0).float(), "label": torch.stack(diagnosis, dim=0).long()}
    return real_out_data



def create_dataset(config, num_process = 8, mode = "train"):
    dataset = EHR_Dataset(config, mode)


    return torch.utils.data.DataLoader(dataset = dataset,
                                       batch_size = int(config.get("train", "batch_size")),
                                       shuffle = True,
                                       num_workers = num_process,
                                       collate_fn = collate_fn)


def init_train_dataset(config):
    return create_dataset(config, 8, "train")


def init_valid_dataset(config):
    return create_dataset(config, 8, "valid")


def init_dataset(config):
    train_dataset = init_train_dataset(config)
    valid_dataset = init_valid_dataset(config)

    return train_dataset, valid_dataset
