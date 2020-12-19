import pickle
from tree.tree_gen import Node

tree = pickle.load(open("json/tree.pkl", "rb"))
id2pos = pickle.load(open("json/id2pos.pkl", "rb"))

for idx in range(500):
    if tree[idx].father != None:
        print(tree[idx].ID, tree[tree[idx].father].ID)
    else:
        print(tree[idx].ID)
    for item in tree[idx].children:
        print(tree[item].ID)
    print("\n\n")
