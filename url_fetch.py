from bs4 import BeautifulSoup
from time import sleep
from urllib.request import build_opener, HTTPCookieProcessor
from urllib.parse import urlencode
from http.cookiejar import CookieJar
import requests
import hashlib

# ヘッダーでユーザーエージェント指定
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0',
}

cookie = dict({
    'name': 'sB',
    'value': 'fp_ipod=0&imsrch_ds=1',
})

req = requests.session()
req.cookies.set(**cookie)


def createURL(url, page_num):

    page_links = list()
    links = list()
    names = list()

    # html = opener.open(url, urlencode(cookie).encode('utf-8'))
    html = req.get(url, headers=headers).text
    soup = BeautifulSoup(html, 'html5lib')

    page_link = soup.find('div', {'id': 'Sp1', 'class': 'mod'}).find_all('a')

    try:

        print("Start make page_urls.")

        for i in range(0, page_num):

            print(i)

            if i == 0:
                # print("add: %s" % page_link[0].attrs['href'])
                page_links.append(page_link[0].attrs['href'])

            else:

                # print("add: %s" % page_link[len(page_link) - 1].attrs['href'])
                page_links.append(page_link[len(page_link) - 1].attrs['href'])

            html = req.get(page_links[len(page_links) - 1], headers=headers).text
            soup = BeautifulSoup(html, 'html5lib')
            page_link = soup.find('div', {'id': 'Sp1', 'class': 'mod'}).find_all('a')

        print("Succeed make page_urls.")

    except Exception as e:
        print("error: %s" % e)


    print("Start collecting images.")

    for j in page_links:

        try:

            sleep(1)
            html = req.get(j, headers=headers).text
            soup = BeautifulSoup(html, 'html5lib')
            datas = soup.find_all('p', {'class': 'tb'})

            for data in datas:

                try:
                    data = data.find('a', {'target': 'imagewin'})
                    links.append(data.attrs['href'].encode('utf-8'))
                    names.append(hashlib.md5(data.attrs['href'].encode('utf-8')).hexdigest())

                    print("%s : createURL successed." % data.attrs['href'].encode('utf-8'))

                except Exception as e:
                    continue

        except Exception as e:
            continue

    print("finish collecting images.")

    Download(links, names)


def Download(links, names):

    print("Start Download image.")

    for (link, name) in zip(links, names):
        try:
            print("access: %s" % link)
            data = req.get(link, headers=headers)
            f = open("images/" + name + ".jpg", "wb")
            f.write(data.content)
            f.close()
            print("imageGet successed.")
            sleep(1)

        except:
            continue

    print("ALL image Saved.")


if __name__ == '__main__':
    createURL(
        'https://search.yahoo.co.jp/image/search?p=akb48+%E3%83%A1%E3%83%B3%E3%83%90%E3%83%BC+%E5%86%99%E7%9C%9F&rkf=1&dim=&imt=&ctype=&imcolor=&imw=0&imh=0&ei=UTF-8&save=0', 10)
