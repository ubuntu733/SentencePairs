#/usr/bin/env python
#coding=utf8
 
import httplib
import md5
import urllib
import random
import json


def geturl(fromLang, toLang, q):
    appid = '20180611000174939'
    secretKey = 'fo6tvOTeutog41p0RTdN'

    httpClient = None
    myurl = '/api/trans/vip/translate'

    salt = random.randint(32768, 65536)

    sign = appid + q + str(salt) + secretKey
    m1 = md5.new()
    m1.update(sign)
    sign = m1.hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
    return myurl

def get_translate(response):
    response = json.loads(response)
    result = []
    for result_dict in response['trans_result']:
        result.append(result_dict['dst'])
    return result

def get_response(url):
    try:
        httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')

        httpClient.request('GET', url)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        tmp = response.read()
        return tmp

    except Exception, e:
        print e
    finally:
        if httpClient:
            httpClient.close()
if __name__=='__main__':
    myurl = geturl(fromLang='zh', toLang='en', q='刘德华的电影')
    result_en = get_translate(get_response(myurl))
    results = []
    for string in result_en:
        myurl = geturl(fromLang='en', toLang='zh', q=string)
        result_zh = get_translate(get_response(myurl))
        results.append(result_zh)
    print(results)
