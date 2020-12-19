import requests
from bs4 import BeautifulSoup as bs
from tree.google import google_translate_EtoC, youdao
from tree.wiki import get_wiki_term
import json
import difflib


html_list = ["https://en.wikipedia.org/wiki/List_of_ICD-9_codes_001%E2%80%93139:_infectious_and_parasitic_diseases",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_140%E2%80%93239:_neoplasms",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_240%E2%80%93279:_endocrine,_nutritional_and_metabolic_diseases,_and_immunity_disorders",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_280%E2%80%93289:_diseases_of_the_blood_and_blood-forming_organs",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_290%E2%80%93319:_mental_disorders",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_320%E2%80%93389:_diseases_of_the_nervous_system_and_sense_organs",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_390%E2%80%93459:_diseases_of_the_circulatory_system",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_460%E2%80%93519:_diseases_of_the_respiratory_system",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_520%E2%80%93579:_diseases_of_the_digestive_system",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_580%E2%80%93629:_diseases_of_the_genitourinary_system",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_630%E2%80%93679:_complications_of_pregnancy,_childbirth,_and_the_puerperium",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_680%E2%80%93709:_diseases_of_the_skin_and_subcutaneous_tissue",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_710%E2%80%93739:_diseases_of_the_musculoskeletal_system_and_connective_tissue",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_740%E2%80%93759:_congenital_anomalies",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_760%E2%80%93779:_certain_conditions_originating_in_the_perinatal_period",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_780%E2%80%93799:_symptoms,_signs,_and_ill-defined_conditions",
             "https://en.wikipedia.org/wiki/List_of_ICD-9_codes_800%E2%80%93999:_injury_and_poisoning"]


kv = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:44.0) Gecko/20100101 Firefox/44.0"}
id2ch_name = {}

fail_ID = []  #{"ID":  , "name" }

def get_neighbor_content(name):
    redir_doc = requests.get("https://baike.baidu.com/search/none?word=%s&pn=0&rn=10&enc=utf8" % name,
                             headers=kv, timeout=10)
    redir_doc.encoding = 'utf-8'
    redir_doc = redir_doc.text
    soup = bs(redir_doc, "html.parser")
    text = soup.get_text()
    text = text.split("\n")
    processed = []
    for line in text:
        if len(line) == 0:
            continue
        if '\u4e00' <= line[0] <= '\u9fff' or line[0].isdigit() == True or line[0] == '（':
            processed.append(line)
    candidate = []
    for line in processed:
        if "_百度百科" in line:
            candidate.append(line[:line.find("_")])
    if len(candidate) == 0:
        return "Redirect Fail"
    max_score = 0
    re_name = None
    for candi in candidate:
        if difflib.SequenceMatcher(None, candi, name).quick_ratio() > max_score:
            max_score = difflib.SequenceMatcher(None, candi, name).quick_ratio()
            re_name = candi
    return re_name

def get_baike_content(ID, name):
    html_doc = requests.get("https://baike.baidu.com/item/" + name, headers=kv, timeout=10)
    html_doc.encoding = 'utf-8'
    html_doc = html_doc.text
    soup = bs(html_doc, "html.parser")
    text = soup.get_text()
    text = text.split("\n")
    processed = []
    for line in text:
        if len(line) == 0:
            continue
        if '\u4e00' <= line[0] <= '\u9fff' or line[0].isdigit() == True or line[0] == '（':
            processed.append(line)

    if "抱歉，您所访问的页面不存在..." in processed:
        rename = get_neighbor_content(name)
        if rename == "Redirect Fail" or rename == None:
            fail_ID.append({"ID": ID, "name": name})
            return "Search Fail"
        return get_baike_content(ID, rename)

    center_word = processed[0][:processed[0].find("_")]
    if "(" in center_word:
        center_word = center_word[:center_word.find("(")]
    if "（" in center_word:
        center_word = center_word[:center_word.find("（")]
    out_doc = []

    catalogue = []
    cata_flag = 0
    cata_label = 0
    for line in processed:
        if line == "基本信息" or line.find(center_word) == 0:
            cata_flag = 0
        if cata_flag == 1 and '\u4e00' <= line[0] <= '\u9fff':
            catalogue.append(line)
        if line == "目录":
            cata_flag = 1
            cata_label = 1
    if cata_label == 1:

        if "临床表现" in catalogue:
            pos = catalogue.index("临床表现")
            flag = 0
            for line in processed:
                if len(line) == 0:
                    continue
                while '\u4e00' > line[0] or line[0] > '\u9fff':
                    line = line[1:]
                    if len(line) == 0:
                        break
                try:
                    if catalogue[pos + 1] in line and line.find(center_word) == 0:
                        break
                except:
                    if len(line) < 10 and flag == 1:
                        break
                if flag == 1:
                    out_doc.append(line)
                if line == center_word + "临床表现":
                    flag = 1
        elif "症状" in catalogue:
            pos = catalogue.index("症状")
            flag = 0
            for line in processed:
                if len(line) == 0:
                    continue
                while '\u4e00' > line[0] or line[0] > '\u9fff':
                    line = line[1:]
                    if len(line) == 0:
                        break
                try:
                    if catalogue[pos + 1] in line and line.find(center_word) == 0:
                        break
                except:
                    if len(line) < 10 and flag == 1:
                        break
                if flag == 1:
                    out_doc.append(line)
                if line == center_word + "症状":
                    flag = 1
        else:
            flag = 0
            for line in processed:
                if line == "编辑":
                    flag = 1
                if line in ["编辑", "锁定", "本词条由国家卫健委权威医学科普项目传播网络平台/百科名医网", "提供内容"] or len(line) < 40:  #过滤掉奇怪的东西
                    continue
                if flag == 1:
                    out_doc.append(line)
                    break
    else:   
        for line in processed:
            if len(line) == 0:
                continue
            while '\u4e00' > line[0] or line[0] > '\u9fff':
                line = line[1:]
                if len(line) == 0:
                    break
            if line.find("症状") == 0 or line.find("临床表现") == 0:
                out_doc.append(line)
    out = ""
    for part in out_doc:
        out += part
    return out


def work(number, web_url):
    dump_data = []
    cnt_all = 0
    cnt_match = 0
    for idx, line in enumerate(get_wiki_term(web_url)):
        id = line[line.find("(") + 1: line.find(")")]
        name = line[line.find(")") + 1 :]
        if name.find("(") != -1:
            name = name[:name.find("(")]
        if name.find(",") != -1:
            name = name[:name.find(",")]
        if len(name) == 0:
            dump_data.append({"ID": id, "content": ""})
            continue
        while name[0] == " ":
            name = name[1:]
            if len(name) == 0:
                break
        if len(name) == 0:
            dump_data.append({"ID": id, "content": ""})
            continue
        ch_name = google_translate_EtoC(name)
        if len(ch_name) == 0:
            dump_data.append({"ID": id, "content": ""})
            continue
        if '\u4e00' > ch_name[0] or ch_name > '\u9fff':
            dump_data.append({"ID": id, "content": ""})
            continue
        print(ch_name)
        content = get_baike_content(id, ch_name)
        cnt_all += 1
        if cnt_all % 100 == 0:
            print("cnt_all=%d" %cnt_all)
        if len(content) < 40 or "本词条" in content:
            dump_data.append({"ID": id, "content": ""})
            continue
        dump_data.append({"ID": id, "content": content})
        cnt_match += 1
    json.dump(dump_data, open("json/" + str(number) + ".json", "w", encoding="utf-8"))

def work_id2ch_name(number, web_url):
    dump_data = []
    cnt_all = 0
    cnt_match = 0
    for idx, line in enumerate(get_wiki_term(web_url)):
        id = line[line.find("(") + 1: line.find(")")]
        name = line[line.find(")") + 1 :]
        if name.find("(") != -1:
            name = name[:name.find("(")]
        if name.find(",") != -1:
            name = name[:name.find(",")]
        if len(name) == 0:
            dump_data.append({"ID": id, "content": ""})
            continue
        while name[0] == " ":
            name = name[1:]
            if len(name) == 0:
                break
        if len(name) == 0:
            continue
        ch_name = google_translate_EtoC(name)
        if len(ch_name) == 0:
            continue
        if '\u4e00' > ch_name[0] or ch_name > '\u9fff':
            continue
        print(ch_name)
        id2ch_name[id] = ch_name


def work_chap6():
    chap_6 = json.load(open("chapter-6.json", "r", encoding="utf-8"))
    dump_data = []
    cnt_all = 0
    cnt_match = 0
    for idx, line in enumerate(chap_6):
        id = line["ID"]
        name = line["name"]
        ch_name = google_translate_EtoC(name)
        if len(ch_name) == 0:
            dump_data.append({"ID": id, "content": ""})
            continue
        if '\u4e00' > ch_name[0] or ch_name > '\u9fff':
            dump_data.append({"ID": id, "content": ""})
            continue
        print(ch_name)
        content = get_baike_content(id, ch_name)
        cnt_all += 1
        if cnt_all % 100 == 0:
            print("cnt_all=%d" %cnt_all)
        if len(content) < 40 or "本词条" in content:
            dump_data.append({"ID": id, "content": ""})
            continue
        dump_data.append({"ID": id, "content": content})
        cnt_match += 1
    json.dump(dump_data, open("json/" + str(5) + ".json", "w", encoding="utf-8"))


if __name__ == "__main__":
    pass