import pickle


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


disease_num2ICD = {0: '780.6', 1: '401.9', 2: '462', 3: '558.9', 4: '786.2', 5: '465.9', 6: '784.1', 7: '789.0', 8: '530.1', 9: '272.4', 10: '414.0',
                   11: '250.0', 12: '276.8', 13: '780.4', 14: '599.0', 15: '787.0', 16: '466.0', 17: '463', 18: '486', 19: '440', 20: '475', 21: '784.0',
                   22: '434.1', 23: '564.0', 24: '786.5', 25: '882', 26: '493', 27: '540.9', 28: '611', 29: '592.9', 30: '724.2', 31: '356.9', 32: '41',
                   33: '845', 34: '785.1', 35: '460', 36: '873', 37: '276.1', 38: '780.5', 39: '496', 40: '436', 41: '585', 42: '571.9', 43: '351.8',
                   44: '477.9', 45: '427.3', 46: '787.3', 47: '285.9', 48: '262', 49: '428', 50: '535', 51: '530.8', 52: '491', 53: '592.1', 54: '574',
                   55: '933', 56: '728.8', 57: '536.8', 58: '518.8', 59: '784.7', 60: '9.1', 61: '592.0', 62: '995.3', 63: '35', 64: '560.9', 65: '786.0',
                   66: '788.0', 67: '600.0', 68: '162.9', 69: '578', 70: '511.9', 71: '575.0', 72: '431', 73: '435.9', 74: '464.0', 75: '577.0', 76: '864',
                   77: '2', 78: '599.7', 79: '412', 80: '351.0', 81: '573', 82: '724.5', 83: '536.3', 84: '269.2', 85: '275.4', 86: '386.1', 87: '274',
                   88: '723.1', 89: '348.2', 90: '591', 91: '413', 92: '410.9', 93: '276.7', 94: '345.9', 95: '722.1', 96: '280.9', 97: '541', 98: '892', 99: '790.6'}   #前100疾病


tree = pickle.load(open("json/tree.pkl", "rb"))
id2pos = pickle.load(open("json/id2pos.pkl", "rb"))

subtree = []
subtree_id2pos = {}
visited_pos = []

def enumerate_node(nownode):
    father_id = len(subtree) - 1
    for child in tree[nownode].children:
        if child not in visited_pos:
            continue
        else:
            subtree.append(Node([], father_id, tree[child].ID, tree[child].name))
            subtree[subtree_id2pos[tree[nownode].ID]].children.append(len(subtree) - 1)
            subtree_id2pos[tree[child].ID] = len(subtree) - 1
            enumerate_node(child)


def geenrate_sub_tree():
    for key in disease_num2ICD.keys():
        pos = id2pos[disease_num2ICD[key]]
        while pos not in visited_pos:
            visited_pos.append(pos)
            pos = tree[pos].father
            if pos == None:
                break
    node = 0
    subtree.append(Node([], None, "S", "root"))
    subtree_id2pos["S"] = 0
    enumerate_node(node)
    pickle.dump(subtree, open("json/sub_tree.pkl", "wb"))
    pickle.dump(subtree_id2pos, open("json/sub_id2pos.pkl", "wb"))


if __name__ == "__main__":
    geenrate_sub_tree()
