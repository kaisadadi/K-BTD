import json
import pickle
from tree.tree_gen import Node
import numpy as np


disease_num2ICD = {0: '780.6', 1: '401.9', 2: '462', 3: '558.9', 4: '786.2', 5: '465.9', 6: '784.1', 7: '789.0', 8: '530.1', 9: '272.4', 10: '414.0',
                   11: '250.0', 12: '276.8', 13: '780.4', 14: '599.0', 15: '787.0', 16: '466.0', 17: '463', 18: '486', 19: '440', 20: '475', 21: '784.0',
                   22: '434.1', 23: '564.0', 24: '786.5', 25: '882', 26: '493', 27: '540.9', 28: '611', 29: '592.9', 30: '724.2', 31: '356.9', 32: '41',
                   33: '845', 34: '785.1', 35: '460', 36: '873', 37: '276.1', 38: '780.5', 39: '496', 40: '436', 41: '585', 42: '571.9', 43: '351.8',
                   44: '477.9', 45: '427.3', 46: '787.3', 47: '285.9', 48: '262', 49: '428', 50: '535', 51: '530.8', 52: '491', 53: '592.1', 54: '574',
                   55: '933', 56: '728.8', 57: '536.8', 58: '518.8', 59: '784.7', 60: '9.1', 61: '592.0', 62: '995.3', 63: '35', 64: '560.9', 65: '786.0',
                   66: '788.0', 67: '600.0', 68: '162.9', 69: '578', 70: '511.9', 71: '575.0', 72: '431', 73: '435.9', 74: '464.0', 75: '577.0', 76: '864',
                   77: '2', 78: '599.7', 79: '412', 80: '351.0', 81: '573', 82: '724.5', 83: '536.3', 84: '269.2', 85: '275.4', 86: '386.1', 87: '274',
                   88: '723.1', 89: '348.2', 90: '591', 91: '413', 92: '410.9', 93: '276.7', 94: '345.9', 95: '722.1', 96: '280.9', 97: '541', 98: '892', 99: '790.6'}   #前100疾病

def get_dis(pre, label):
    #pre和label都是整数的疾病编码
    #按照各自向上爬，取交集即可
    global tree, id2pos
    pre = id2pos[disease_num2ICD[pre]]
    label = id2pos[disease_num2ICD[label]]
    pre_s = set()
    lab_s = set()
    step_pre = 0
    step_label = 0
    while True:
        #先pre动，再label动
        pre_f = tree[pre].father
        pre = pre_f
        pre_s.add(pre_f)
        step_pre += 1
        if len(pre_s & lab_s) != 0:
            return step_pre + step_label
        #再label动
        lab_f = tree[label].father
        label = lab_f
        lab_s.add(lab_f)
        step_label += 1
        if len(pre_s & lab_s) != 0:
            return step_pre + step_label


if __name__ == "__main__":
    tree = pickle.load(open("../tree/json/sub_tree.pkl", "rb"))
    id2pos = pickle.load(open("../tree/json/sub_id2pos.pkl", "rb"))
    D = np.zeros([100, 100], np.float32)
    for a in range(0, 100):
        for b in range(a+1, 10):
            D[a, b] = get_dis(a, b)
            D[b, a] = D[a, b]

    SVM_data = json.load(open("../tree/json/TextCNN.json", "r"))
    sum_all = 0
    for idx in range(len(SVM_data)):
        sum1 = 0
        for item1 in SVM_data[idx]["std"]:
            min_val = 8
            for item2 in SVM_data[idx]["pre"]:
                min_val = min([min_val, D[int(item1), int(item2)]])
            sum1 += min_val
        sum2 = 0
        for item1 in SVM_data[idx]["pre"]:
            min_val = 8
            for item2 in SVM_data[idx]["std"]:
                min_val = min([min_val, D[int(item1), int(item2)]])
            sum2 += min_val
        if len(SVM_data[idx]["pre"]) != 0:
            sum_all += sum1 / len(SVM_data[idx]["std"]) + sum2 / len(SVM_data[idx]["pre"])
        else:
            sum_all += sum1 / len(SVM_data[idx]["std"])
    print(sum_all / len(SVM_data))
