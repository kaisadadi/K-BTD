#生成mask的数据

import json
import random

syp_dict = json.load(open("../tree/json/syp_dict.json", "r"))


def masked_gen(ratio):
    new_data = {}
    for key in syp_dict.keys():
        temp = syp_dict[key]
        length = len(temp["syptom"])
        if length > 10:
            pos = random.randint(0, int((1 - ratio) * length - 1))
            new_syp = temp["syptom"][:pos] + temp["syptom"][pos + int(ratio * length):]
            temp["syptom"] = new_syp
            new_data[key] = temp
        else:
            new_data[key] = temp
    json.dump(new_data, open("../tree/json/syp_dict_%s.json" %str(ratio), "w"))

def shullfed_gen(ratio):
    random_data = []
    shuffle_key = []
    for key in syp_dict.keys():
        temp = syp_dict[key]
        if random.randint(1, 10) <= int(ratio * 10):
            random_data.append(temp)
            shuffle_key.append(key)
    for key in shuffle_key:
        syp_dict[key] = random_data[random.randint(0, len(random_data) - 1)]
    json.dump(syp_dict, open("../tree/json/syp_dict_s_%s.json" % str(ratio), "w"))

def replaced_gen(ratio):
    global random_text
    for key in syp_dict.keys():
        temp = syp_dict[key]['syptom']
        if random.randint(1, 10) <= int(ratio * 10):
            pos = random.randint(0, len(random_text) - len(temp) - 1)
            syp_dict[key]['syptom'] = random_text[pos : pos + len(temp)]
    json.dump(syp_dict, open("../tree/json/syp_dict_r_%s.json" % str(ratio), "w"))

if __name__ == "__main__":
    random_text = ""
    f = open("../tree/random-text.txt", "r", encoding="gbk")
    for line in f.readlines():
        line = line[:-1]
        random_text += line
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        #masked_gen(ratio)
        #shullfed_gen(ratio)
        #replaced_gen(ratio)
        pass
    ratio = 0.5
    data = json.load(open("../tree/json/syp_dict_r_%s.json" % str(ratio), "r"))
    print(data)
