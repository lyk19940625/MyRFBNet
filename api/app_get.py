# --------------------------------------------------------
# object detection restful api
# author:wang lei
# update time:2018-9-21
#  --------------------------------------------------------
import sys

sys.path.append('../')
import cv2
from flask import Flask, jsonify
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
from detector import ObjDetector

app = Flask(__name__)
api = Api(app)

parser = RequestParser()
parser.add_argument('imgpth', type=str, location=['args'], help='image local path')

detector = ObjDetector()


class DetectorServer(Resource):
    def get(self):
        try:
            args = parser.parse_args()
            imgpth = args['imgpth']
            image = cv2.imread(imgpth)
            bboxs = detector(image)

            data = list()
            for label, value in bboxs.items():
                for v in value:
                    box = dict()
                    box['x1'] = str(round(v[0]))
                    box['y1'] = str(round(v[1]))
                    box['x2'] = str(round(v[2]))
                    box['y2'] = str(round(v[3]))
                    box['score'] = str(round(v[4], 2))
                    box['label'] = label
                    data.append(box)
            rest = dict()
            rest['rtnCode'] = '000000'
            rest['rtnMsg'] = 'success'
            rest['rtnData'] = data
            return jsonify(rest)
        except Exception as e:
            return jsonify({'rtnCode': '100000', 'rtnMsg': str(e)})


api.add_resource(DetectorServer, '/objectDetection/')


class HelloWorld(Resource):
    def get(self):
        return {'Object Detection Server API'}


api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(host='192.168.9.234', debug=False, threaded=True, port=5000)
