import json

print("start loading")
f = open("embedding.json", "r")
embed_dict = json.load(f)

print("load embedding done")

pre_embedding = json.load(open("word_embedding.json", "r"))
new_material = json.load(open("syp_dict.json", "r"))
zero_vec = []
for a in range(200):
    zero_vec.append(0.00)
import jieba

for key in new_material.keys():
    term = jieba.cut(new_material[key]['name'])
    syp = jieba.cut(new_material[key]['syptom'])
    for data in [term, syp]:
        for word in data:
            if word not in pre_embedding.keys():
                if word in embed_dict.keys():
                    pre_embedding[word] = embed_dict[word]
                else:
                    pre_embedding[word] = zero_vec

json.dump(pre_embedding, open("all_word_embedding.json", "w", encoding="utf-8"), indent=2)