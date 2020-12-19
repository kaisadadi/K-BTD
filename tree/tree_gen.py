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

if __name__ == "__main__":
    id2pos = {}
    tree = []
    #root
    tree.append(Node([], None, "S", "root"))
    id2pos["S"] = 0
    #ICD-9 section
    dalei = ["传染病和寄生虫病", "肿瘤", "内分泌，营养和代谢疾病，以及免疫紊乱",
         "血液和造血器官疾病", "精神疾病", "神经系统和感觉器官疾病",
         "循环系统疾病", "呼吸系统疾病", "消化系统疾病", "泌尿生殖系统疾病",
         "妊娠，分娩和产褥期的并发症", "皮肤和皮下组织疾病",
         "肌肉骨骼系统和结缔组织疾病", "先天性异常", "围产期疾病", "症状，体征和不明确的情况",
         "受伤和中毒"]

    for idx, item in enumerate(dalei):
        tree.append(Node([], 0, "S_%d" %idx, item))
        tree[0].children.append(len(tree) - 1)
        id2pos["S_%d" %idx] = len(tree) - 1

    chap = []
    for i in range(17):
        chap.append([])

    chap[0] = [{"name":"细菌和结核感染","low": 2, "high": 42},
           {"name": "病毒感染", "low": 42, "high": 80},
           {"name": "立克次体病和其他节肢动物传播的疾病", "low": 80, "high": 90},
           {"name": "性病", "low": 90, "high": 100},
           {"name": "螺旋体病", "low": 100, "high": 105},
           {"name": "真菌感染", "low": 110, "high": 119},
           {"name": "蠕虫病", "low": 120, "high": 130},
           {"name": "其他传染病和寄生虫病", "low": 130, "high": 136},
           {"name": "感染后遗症", "low": 136, "high": 140}]

    chap[1] = [{"name":"唇，口腔和咽部恶性肿瘤","low": 140, "high": 150},
           {"name": "消化器官和腹膜恶性肿瘤", "low": 150, "high": 160},
           {"name": "呼吸道和胸腔内器官恶性肿瘤", "low": 160, "high": 166},
           {"name": "骨，结缔组织，皮肤和乳房恶性肿瘤", "low": 170, "high": 176},
           {"name": "卡波西氏肉瘤", "low": 176, "high": 177},
           {"name": "泌尿生殖器官恶性肿瘤", "low": 179, "high": 190},
           {"name": "其他和未指明部位的恶性肿瘤", "low": 190, "high": 200},
           {"name": "淋巴和造血组织的恶性肿瘤", "low": 200, "high": 209},
           {"name": "神经内分泌肿瘤", "low": 209, "high": 210},
           {"name": "良性肿瘤", "low": 210, "high": 230},
           {"name": "原位癌", "low": 230, "high": 235},
           {"name": "行为不确定的肿瘤", "low": 235, "high": 239},
           {"name": "未明确性质的肿瘤", "low": 239, "high": 240}]

    chap[2] = [{"name":"甲状腺疾病","low": 240, "high": 247},
           {"name": "其他内分泌腺疾病", "low": 249, "high": 260},
           {"name": "营养不良", "low": 260, "high": 270},
           {"name": "其他代谢和免疫疾病", "low": 270, "high": 280}]

    chap[3] = [{"name":"贫血","low": 280, "high": 286},
           {"name": "凝血功能障碍和出血", "low": 286, "high": 288},
           {"name": "其他血液疾病", "low": 288, "high": 290}]

    chap[4] = [{"name":"精神病","low": 290, "high": 300},
           {"name": "神经症，人格障碍和其他非精神病性精神障碍", "low": 300, "high": 317},
           {"name": "精神发育迟滞", "low": 317, "high": 320}]

    chap[5] = [{"name":"中枢神经系统的炎症性疾病","low": 320, "high": 328},
           {"name": "中枢神经系统的遗传性和退行性疾病", "low": 330, "high": 338},
           {"name": "疼痛", "low": 338, "high": 339},
           {"name": "其他头痛综合症", "low": 339, "high": 340},
           {"name": "其他中枢神经系统疾病", "low": 340, "high": 350},
           {"name": "周围神经系统疾病", "low": 350, "high": 360},
           {"name": "眼睛和眼睑和泪腺的障碍", "low": 360, "high": 380},
           {"name": "耳朵和乳突疾病", "low": 380, "high": 390}]

    chap[6] = [{"name":"急性风湿热","low": 390, "high": 393},
           {"name": "慢性风湿性心脏病", "low": 393, "high": 399},
           {"name": "高血压病", "low": 401, "high": 406},
           {"name": "缺血性心脏病", "low": 410, "high": 415},
           {"name": "肺循环疾病", "low": 415, "high": 418},
           {"name": "其他形式的心脏病", "low": 420, "high": 430},
           {"name": "脑血管病", "low": 430, "high": 439},
           {"name": "动脉，小动脉和毛细血管疾病", "low": 440, "high": 450},
           {"name": "静脉和淋巴管疾病以及其他循环系统疾病", "low": 451, "high": 460}]

    chap[7] = [{"name":"急性呼吸道感染","low": 460, "high": 467},
           {"name": "其他上呼吸道疾病", "low": 470, "high": 479},
           {"name": "肺炎和流感", "low": 480, "high": 489},
           {"name": "慢性阻塞性肺病和相关疾病", "low": 490, "high": 497},
           {"name": "尘肺病和其他肺部疾病", "low": 500, "high": 509},
           {"name": "其他呼吸系统疾病", "low": 510, "high": 520},]

    chap[8] = [{"name":"口腔，唾液腺和颌骨疾病","low": 520, "high": 530},
           {"name": "食道，胃和十二指肠疾病", "low": 530, "high": 540},
           {"name": "阑尾炎", "low": 540, "high": 544},
           {"name": "腹部疝", "low": 550, "high": 554},
           {"name": "非传染性肠炎和结肠炎", "low": 555, "high": 559},
           {"name": "其他肠道和腹膜疾病", "low": 560, "high": 570},
           {"name": "其他消化系统疾病", "low": 570, "high": 580}]

    chap[9] = [{"name":"肾炎，肾病综合征和肾病","low": 580, "high": 590},
               {"name": "泌尿系统其他疾病", "low": 590, "high": 600},
               {"name": "男性生殖器官疾病", "low": 600, "high": 609},
               {"name": "乳房疾病", "low": 610, "high": 613},
               {"name": "女性盆腔器官炎症性疾病", "low": 614, "high": 617},
               {"name": "其他女性生殖道疾病", "low": 617, "high": 630}]


    chap[10] = [{"name":"异位和磨牙妊娠","low": 630, "high": 634},
           {"name": "流产", "low": 634, "high": 640},
           {"name": "妊娠并发症", "low": 640, "high": 650},
           {"name": "正常分娩，以及其他妊娠，分娩和分娩护理指征", "low": 650, "high": 660},
           {"name": "分娩并发症", "low": 660, "high": 670},
           {"name": "产褥期的并发症", "low": 670, "high": 678},
           {"name": "其他孕产妇和胎儿并发症", "low": 678, "high": 680}]

    chap[11] = [{"name":"皮肤和皮下组织感染","low": 680, "high": 687},
           {"name": "皮肤和皮下组织的其他炎症", "low": 690, "high": 699},
           {"name": "皮肤和皮下组织的其他疾病", "low": 700, "high": 710}]

    chap[12] = [{"name":"关节病和相关疾病","low": 710, "high": 720},
           {"name": "颈痛，腰痛，背痛", "low": 720, "high": 725},
           {"name": "风湿病，不包括背部", "low": 725, "high": 730},
           {"name": "骨病，软骨病和后天性肌肉骨骼畸形", "low": 730, "high": 740}]

    chap[13] = [{"name":"先天性神经系统异常","low": 740, "high": 743},
           {"name": "先天性眼耳脸颈异常", "low": 743, "high": 745},
           {"name": "先天性循环系统异常", "low": 745, "high": 748},
           {"name": "先天性呼吸系统异常", "low": 748, "high": 749},
           {"name": "先天性消化系统异常", "low": 749, "high": 752},
           {"name": "先天性生殖器官异常", "low": 752, "high": 753},
           {"name": "先天性泌尿系统异常", "low": 753, "high": 754},
           {"name": "先天性肌肉骨骼异常", "low": 754, "high": 757},
           {"name": "先天性染色体异常", "low": 758, "high": 759},
           {"name": "先天性其他异常", "low": 759, "high": 760}]

    chap[14] = [{"name":"围产期发病与死亡","low": 760, "high": 764},
           {"name": "围产期其他疾病", "low": 764, "high": 780}]


    chap[15] = [{"name":"症状","low": 780, "high": 790},
           {"name": "非特异性异常", "low": 790, "high": 797},
           {"name": "发病及死亡的不确定性异常", "low": 797, "high": 800}]


    chap[16] = [{"name":"颅骨骨折","low": 800, "high": 805},
           {"name": "颈部和躯干骨折", "low": 805, "high": 810},
           {"name": "上肢骨折", "low": 810, "high": 820},
           {"name": "下肢骨折", "low": 820, "high": 830},
           {"name": "脱臼", "low": 830, "high": 840},
           {"name": "关节和邻近肌肉的扭伤和拉伤", "low": 840, "high": 849},
           {"name": "颅内损伤，不包括颅骨骨折", "low": 850, "high": 855},
           {"name": "胸部，腹部和骨盆内部损伤", "low": 860, "high": 870},
           {"name": "头部，颈部和躯干的开放性伤口", "low": 870, "high": 880},
           {"name": "上肢开放性伤口", "low": 880, "high": 888},
           {"name": "下肢开放性伤口", "low": 890, "high": 898},
           {"name": "血管受伤", "low": 900, "high": 905},
           {"name": "伤害和中毒的后期影响", "low": 905, "high": 910},
            {"name": "表面伤害", "low": 910, "high": 920},
            {"name": "皮肤表面挫伤", "low": 920, "high": 925},
            {"name": "挤压伤", "low": 925, "high": 930},
            {"name": "异物通过人体口进入的影响", "low": 930, "high": 940},
            {"name": "烧伤", "low": 940, "high": 950},
            {"name": "神经和脊髓损伤", "low": 950, "high": 958},
            {"name": "某些创伤性并发症和未明确的损伤", "low": 958, "high": 960},
            {"name": "药物，药物和生物物质中毒", "low": 960, "high": 980},
            {"name": "非药物性中毒", "low": 980, "high": 990},
            {"name": "外部原因造错的损伤", "low": 990, "high": 996},
            {"name": "手术和医疗并发症", "low": 996, "high": 1000}]




    for idx_chap in range(17):
        for idx_name in range(len(chap[idx_chap])):
            tree.append(Node([], id2pos["S_%d" %idx_chap], "S_%d_%d" %(idx_chap, idx_name), chap[idx_chap][idx_name]["name"]))
            id2pos["S_%d_%d" %(idx_chap, idx_name)] = len(tree) - 1
            tree[id2pos["S_%d" %idx_chap]].children.append(len(tree) - 1)


    ICD_set = json.load(open("json/ICD.json", "r"))
    for idx_chap in range(17):
        for idx_name in range(len(chap[idx_chap])):
            for num in range(chap[idx_chap][idx_name]["low"], chap[idx_chap][idx_name]["high"]):

                if str(num) in ICD_set.keys():
                    tree.append(Node([], id2pos["S_%d_%d" %(idx_chap, idx_name)], str(num), str(num)))
                    id2pos[str(num)] = len(tree) - 1
                    tree[id2pos["S_%d_%d" %(idx_chap, idx_name)]].children.append(len(tree) - 1)

                for decimal in range(10):
                    if str(num) not in ICD_set.keys():
                        break
                    if str(num) + "." + str(decimal) in ICD_set.keys():
                        tree.append(Node([], id2pos[str(num)], str(num) + "." + str(decimal), str(num) + "." + str(decimal)))
                        id2pos[str(num) + "." + str(decimal)] = len(tree) - 1
                        tree[id2pos[str(num)]].children.append(len(tree) - 1)

    #存储tree
    pickle.dump(tree, open("json/tree.pkl", "wb"))
    pickle.dump(id2pos, open("json/id2pos.pkl", "wb"))
