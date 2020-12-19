import json
import pandas as pd

df = pd.read_excel("../tree/json/human_note.xlsx")

#print(df.keys())
#print(len(df))
#print(df.iloc[0])

syp_dict = {}

def xlsx2json():
    for idx in range(len(df)):
        if df.iloc[idx][1] not in syp_dict.keys():
            name = df.iloc[idx][0]
            diagnosis = df.iloc[idx][3]
            if isinstance(diagnosis, float):
                diagnosis = ""
            diagnosis.replace('\n', '')
            syp_dict[df.iloc[idx][1]] = {"name": name, "syptom": diagnosis}
        else:
            continue
    json.dump(syp_dict, open("../tree/json/syp_dict.json", "w"))

def test_json():
    syp_dict = json.load(open("../tree/json/syp_dict.json", "r"))
    print(syp_dict["S_5_2"])



xlsx2json()
#test_json()



