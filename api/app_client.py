# --------------------------------------------------------
# object detection restful api
# author:wang lei
# update time:2018-9-21
#  ---------------------------------------------------
# encoding: utf-8
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import requests
import re
import time
time1=time.time()

from requests import put, get, post

#html=put('http://localhost:5000/todo1', data={'data': 'Remember the milk'}).json()
#print html

get_html=get('http://192.168.9.234:5000/objectDetection/?imgpth=/home/qingqing/mlproject/MyRFBNet/images/000215.jpg').json()
print(get_html)


#url="http://192.168.9.234:4203/objectDetection/"

#parms = {
#    'imgpth': '/home/qingqing/mlproject/MyRFBNet/images/000215.jpg'  # 发送给服务器的内容
#}
 
#headers = {
#    'User-agent': 'none/ofyourbusiness',
#    'Spam': 'Eggs'
#}

#res = post(url, data=parms, headers=headers).json()
#print(res)
