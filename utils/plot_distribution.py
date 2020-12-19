import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

train_data = pickle.load(open("../train.pkl", "rb"))

statis = np.zeros([32])

for item in train_data:
    statis[len(item["急诊诊断"])] += 1

statis = statis / np.sum(statis)

matplotlib.rcParams['font.style'] = 'italic'
matplotlib.rcParams['font.size'] = 12
plt.xlabel("Number of diagnosis")
plt.ylabel("% of admissions")
plt.xticks(np.linspace(1, 10, 10, endpoint=True))
plt.bar(np.linspace(1, 10, 10, endpoint=True), statis[1:11])
plt.savefig("../3.png", dpi=500)
plt.show()
