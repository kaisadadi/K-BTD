'''import pickle
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

f = open("/home/wke18/Changgeng-Hospital/Data/train.pkl", "rb")
data = pickle.load(f)

disease_num2ICD = json.load(open("/home/wke18/Changgeng-Hospital/Data/disease_num2ICD.json", "r"))
ICD2pos = pickle.load(open("/home/wke18/Changgeng-Hospital/Code/tree/json/id2pos.pkl", "rb"))
mytree = pickle.load(open("/home/wke18/Changgeng-Hospital/Code/tree/json/tree.pkl", "rb"))


for idx in range(50):
    print(data[idx]['急诊诊断'])
    x = []
    for disease in data[idx]['急诊诊断']:
        x.append(ICD2pos[disease_num2ICD[str(disease)]])
        father = []
        temp = x[-1]
        while temp != None:
            father.append(temp)
            temp = mytree[temp].father
        print(father[::-1])
    print("-------------------------------------")'''

import json
data = json.load(open("../tree/json/syp_dict.json", "r"))

print(data["S_2"])
