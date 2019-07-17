# --------------------------------------------------------
# web service
# aurthor wanglei
# update time:2018-9-12
# --------------------------------------------------------

from __future__ import print_function

import sys

sys.path.append('../')
from spyne import Application
from spyne import rpc
from spyne import ServiceBase
from spyne import Iterable, Unicode
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication
from detector import ObjDetector
import cv2

detector = ObjDetector()


class ObjDetectorService(ServiceBase):
    @rpc(Unicode, _returns=Iterable(Unicode))
    def img_obj(self, image_file):
        img = cv2.imread(image_file)
        bboxs = detector(img)
        for label, value in bboxs.items():
            for v in value:
                x1 = str(round(v[0]))
                y1 = str(round(v[1]))
                x2 = str(round(v[2]))
                y2 = str(round(v[3]))
                score = str(round(v[4], 2))
                label = label
                ret = x1 + ',' + y1 + ',' + x2 + ',' + y2 + ',' + score + ',' + label
                yield ret


# step2: Glue the service definition, input and output protocols
soap_app = Application([ObjDetectorService], 'spyne.ObjDetector.service.soap',
                       in_protocol=Soap11(validator='lxml'),
                       out_protocol=Soap11())

# step3: Wrap the Spyne application with its wsgi wrapper
wsgi_app = WsgiApplication(soap_app)

if __name__ == '__main__':
    import logging

    from wsgiref.simple_server import make_server

    # configure the python logger to show debugging output
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('spyne.protocol.xml').setLevel(logging.DEBUG)

    logging.info("listening to http://192.168.9.234:6900")
    logging.info("wsdl is at: http://192.168.9.234:6900/?wsdl")

    # step4:Deploying the service using Soap via Wsgi
    # register the WSGI application as the handler to the wsgi server, and run the http server
    server = make_server('192.168.9.234', 6900, wsgi_app)
    server.serve_forever()
