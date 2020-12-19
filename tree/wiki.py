import requests
from bs4 import BeautifulSoup as bs
import json


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


def get_wiki_term(html_list):
    for url in [html_list]:
        html_doc = requests.get(url).text
        soup = bs(html_doc, "html.parser")
        text = soup.get_text()
        text = text.split("\n")
        for line in text:
            if len(line) == 0:
                continue
            if line[0] == '(':
                try:
                    x = int(line[1])
                    yield(line)
                except:
                    continue


def get_more_wiki(html_list):
    out_file = []
    jilu = {}
    for url in html_list:
        html_doc = requests.get(url).text
        soup = bs(html_doc, "html.parser")
        text = soup.get_text()
        text = text.split("\n")
        begin_flag = 0
        number = None
        for line in text:
            if len(line) == 0:
                continue
            if line == "BILLABLE" or line == "NON-BILLABLE":
                continue
            if line[0].isdigit() == True:
                begin_flag = 1
            if begin_flag == 1:
                if line[0].isdigit() == True:
                    if number == None and line not in jilu.keys():
                        jilu[line] = 0
                        number = line
                if line[0].isalpha() == True:
                    if number != None:
                        out_file.append({"ID": number, "name": line})
                        number = None
    for record in out_file:
        print(record["ID"], record["name"])
    json.dump(out_file, open("chap6.json", "w", encoding="utf-8"))



