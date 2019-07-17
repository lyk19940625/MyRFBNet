#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from suds.client import Client

wsdl_url = "http://192.168.9.234:6900/?wsdl"


def img_obj_test(url, image_file):
    client = Client(url)
    client.service.img_obj(image_file)
    req = str(client.last_sent())
    response = str(client.last_received())
    #print(req)
    print(response)


if __name__ == '__main__':
    img_file = '/home/qingqing/mlproject/MyRFBNet/images/000215.jpg'
    img_obj_test(wsdl_url, img_file)

