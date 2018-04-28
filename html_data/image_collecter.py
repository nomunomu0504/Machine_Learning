from time import sleep
import requests

# ヘッダーでユーザーエージェント指定
headers = {'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}


def make_links(address, count, counter):
    links = []
    for i in range(0, count):
        links.append(address + str(i) + '/img/p')
    Download(links, counter)


def Download(links, counter):
    for (link, index) in zip(links, range(0, len(links))):
        data = requests.get(link, headers=headers)
        f = open("images/" + str(counter) + "_" + str(index) + ".jpg", "wb")
        f.write(data.content)
        f.close()
        print("%s_imageGet successed." % index)
        sleep(1)

if __name__ == '__main__':
    # make_links('http://a.scn.jp/priv/RDwdYhaAC/photos/', 690, 1) #520
    make_links('http://a.scn.jp/priv/XHekXjxAC/photos', 977, 2)
