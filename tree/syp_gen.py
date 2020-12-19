import json
import pickle

#syp_dict = {"ID" : {"name": , "syptom": }}

from tree.tree_gen import Node

tree = pickle.load(open("json/tree.pkl", "rb"))
all_id2ch_name = json.load(open("json/all_id2ch_name.json", "r"))
syp_dict = {}

for node in tree:
    if "S" in node.ID:
        syp_dict[node.ID] = {"name": node.name, "syptom": ""}
    else:
        if node.ID in all_id2ch_name.keys():
            syp_dict[node.ID] = {"name": all_id2ch_name[node.ID], "syptom": ""}
        else:
            syp_dict[node.ID] = {"name": "未知疾病", "syptom": ""}

cnt = 0
for idx in range(17):
    f = open("json/%d.txt" %idx, "r", encoding="utf-8-sig")
    for line in f.readlines():
        line = line.split(" ")
        if line[1] != '\n':
            line[0] = line[0].lstrip('0')
            if line[0].find(".") != -1:
                pos = line[0].find(".") + 2
                line[0] = line[0][:pos]
            try:
                syp_dict[line[0].lstrip('0')]["syptom"] = line[1]
            except:
                print(line[0].lstrip('0'))

json.dump(syp_dict, open("json/syp_dict.json", "w"))
