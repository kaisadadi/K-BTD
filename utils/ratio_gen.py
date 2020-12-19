#用于求subtree每个节点各个儿子的分布情况

import pickle
import json

class Node(object):
    def __init__(self, children, father, ID, name):
        self.children = children
        self.father = father
        self.name = name
        self.ID = ID

    def get_children(self):
        return self.children

    def get_father(self):
        return self.father

    def get_name(self):
        return self.name

    def add_children(self, child):
        self.children.append(child)

    def show(self):
        print(self.ID, self.name, self.children, self.father)

subtree = pickle.load(open("/home/wke18/Changgeng-Hospital/Code/tree/json/sub_tree.pkl", "rb"))
disease_num2ICD = json.load(open("/home/wke18/Changgeng-Hospital/Data/disease_num2ICD.json", "r"))
ICD2pos = pickle.load(open("/home/wke18/Changgeng-Hospital/Code/tree/json/sub_id2pos.pkl", "rb"))

f = open("/home/wke18/Changgeng-Hospital/Data/train.pkl", "rb")
train_data = pickle.load(f)

class Node_statis(object):
    def __init__(self, ratio, ID):
        self.ratio = ratio
        self.ID = ID

    def show(self):
        print(self.ID, self.ratio)


sub_tree_statis = []
#先根据subtree初始化
for node in subtree:
    sub_tree_statis.append(Node_statis([0] * len(node.children), node.ID))

#再根据训练集进行统计
for item in train_data:
    diagnosis = item["急诊诊断"]
    for disease in diagnosis:
        node = ICD2pos[disease_num2ICD[str(disease)]]
        while subtree[node].name != "root":
            father = subtree[node].father
            for idx, son in enumerate(subtree[father].children):
                if son == node:
                    sub_tree_statis[father].ratio[idx] += 1
                    break
            node = subtree[node].father

pickle.dump(sub_tree_statis, open("/home/wke18/Changgeng-Hospital/Code/tree/json/sub_tree_statis.pkl", "wb"))

''''
sub_tree_statis = pickle.load(open("/home/wke18/Changgeng-Hospital/Code/tree/json/sub_tree_statis.pkl", "rb"))
number = 2
subtree[number].show()
sub_tree_statis[number].show()


if __name__ == "__main__":
    pass'''
