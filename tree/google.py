import requests
from bs4 import BeautifulSoup
import urllib


def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except:
        print("Get HTML Text Failed!")
        return 0


def google_translate_EtoC(to_translate, from_language="en", to_language="ch-CN"):
    base_url = "https://translate.google.cn/m?hl={}&sl={}&ie=UTF-8&q={}"
    url = base_url.format(to_language, from_language, to_translate)

    html = getHTMLText(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")

    try:
        result = soup.find_all("div", {"class": "t0"})[0].text
    except:
        print("Translation Failed!")
        result = ""

    return result


def google_translate_CtoE(to_translate, from_language="ch-CN", to_language="en"):
    base_url = "https://translate.google.cn/m?hl={}&sl={}&ie=UTF-8&q={}"
    url = base_url.format(to_language, from_language, to_translate)

    html = getHTMLText(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")

    try:
        result = soup.find_all("div", {"class": "t0"})[0].text
    except:
        print("Translation Failed!")
        result = ""

    return result


def youdao(content):
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'

    headers = {
        "User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:59.0) Gecko/20100101 Firefox/59.0'
    }
    format_data = {
        'i': content,
        'from': 'AUTO',
        'to': 'AUTO',
        'smartresult': 'dict',
        'client': 'fanyideskweb',
        'salt': '1526368137702',
        'sign': 'f0cd13ef1919531ec9a66516ceb261a5',
        'doctype': 'json',
        'version': '2.1',
        'keyfrom': 'fanyi.web',
        'action': 'FY_BY_REALTIME',
        'typoResult': 'false'
    }

    format_data = urllib.parse.urlencode(format_data).encode("utf-8")

    request = urllib.request.Request(url, data=format_data, headers=headers)

    response = urllib.request.urlopen(request)

    content = response.read()
    content = eval(content)
    ret = content["translateResult"][0][0]['tgt']

    return ret

def main():
    while True:
        inp = int(input("Chinese to Englisth is 1, English to Chinese is 2:    "))
        if inp == 1:
            words = input("请输入中文:    ")
            print(google_translate_CtoE(words))
        else:
            words = input("Please input English:    ")
            print(google_translate_EtoC(words))

if __name__ == "__main__":
    main()