# --------------------------------------------------------
# object detection restful api
# author:wang lei
# update time:2018-9-21
#  --------------------------------------------------------
import os, sys

sys.path.append('../')
import cv2
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
from detector import ObjDetector

app = Flask(__name__)
api = Api(app)

parser = RequestParser()
parser.add_argument('imgpth', type=str, required=True, help='image local path')
#parser.add_argument('rtspflow', type=str, location=['args'], help='rtsp video flow address')
#parser.add_argument('detecflag', type=str, location=['args'], help='detection flag')


detector = ObjDetector()


class DetectorServer(Resource):
    def post(self):
        try:
            imgpth = upload()
            #args = parser.parse_args()
            #imgpth = args['imgpth']
            image = cv2.imread(imgpth)
            if os.path.exists(imgpth):
                os.remove(imgpth)
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

def upload():
    fname = request.files.get('file')  #获取上传的文件
    if fname:
        #t = time.strftime('%Y%m%d%H%M%S')
        new_fname = r'/home/qingqing/mlproject/MyRFBNet/file_temp/' + fname.filename
        fname.save(new_fname)  #保存文件到指定路径
        #return '{"code": "ok"}', new_fname
        return new_fname
    else:
        return '{"msg": "请上传文件！"}'




class HelloWorld(Resource):
    def get(self):
        return {'Object Detection Server API'}


api.add_resource(HelloWorld, '/')


if __name__ == '__main__':
    app.run(host='192.168.9.234', debug=False, threaded=True, port=4203)
