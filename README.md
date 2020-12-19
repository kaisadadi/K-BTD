### Automatic Diagnosis with Knowledge-based Tree Decoding

This is the public code for "Automatic Diagnosis with Knowledge-based Tree Decoding" to be published in IJCAI-2020.

As the dataset is in Chinese, some part in the code might contain Chinese characters. If you have problem understanding, please refer to *translate.google.com*.

#### 1. Code Structure

The code is organized as follows:

**config:**

The config folder stores all config files. The config file determines the settings and parameters of model train/eval and test. Each config file has four sections, namely "train", "data", "model" and "output", which control the settings respectively. The filed in each section is pretty straight-forward, and you can understand the meaning easily. 

**config_reader:**

The config_reader folder consists of mainly the *ConfigParser* class, which serves to interpret the config files. The logic of config setting is here.

**model:**

The model folder consists of model files. The model in this paper and some baselines are provided. 

**reader:**

The reader folder consists of the dataset class. In this work, we utilize the original *torch.utils.data.DataLoader* class to realize the function of loading data. To keep consistency, our dataset class extends the original *torch.utils.data.Dataset*, and use the *collate_fn* function as data formatter. 

**tree:**

The disease tree and external knowledge are key in our model. This folder consists of all codes related to tree structure generation and knowledge extraction. The tree structure is crawled from wikipedia (as can be seen in *tree/wiki.py* and *tree/tree_gen.py*). The external knowledge is crawled as in *google.py* and *baidubaike.py*.

As more and more researches are conducted in the medical field, there are already public versions of ICD tree structure. If you want to get the structure, do not waste your time in the tree generation process.

In *tree/json*, some examples of nodes and external knowledge can be seen.

**utils:**

This folder contains functions of various purpose, such as accuracy function, semantic distance, and the random shuffle/mask/replacement in ablation analysis.

**train.py & eval.py:**

This two file are entrance of training and evaluation. Two important settings must be provided in the command line, which is the the relative path of config file(--config) and the gpu settings(--gpu).

e.g. python3 train.py --config xx/xxx.config  --gpu 0,1,2



#### 2. Model

In the *model* folder, different versions of our proposed model are provided.

**version 1: model/MODEL_V1.py**

In this version, we use an LSTM to serve as the Judge Net, and no Fusion Net involved. Terms(node name) and symptoms(node knowledge) for each node are encoded separately. 

**version 2: model/MODEL_V2.py**

In this version, we have the Judge Net as described in the paper, but no Fusion Net involved. Teacher forcing is added to boost the training process.

**version 3: model/MODEL_V3.py**

This is the standard version as described in the paper.



If you have any questions regarding this paper or the code here, please contact **wangke18@mails\.tsinghua.edu.cn**, or **bruuceke@gmail\.com**, as the school email will no-longer be accessible after graduation. 