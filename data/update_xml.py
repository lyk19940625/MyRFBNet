# ---------------------------
# update long name to short name
# ---------------------------
import os

from utils import update_obj_name

xml_root = 'E:/guangy/Annotations/'
xml_name = os.path.join(xml_root, '%s')
for file in os.listdir(xml_root):
    # print(file)
    name = xml_name % file
    update_obj_name(name)
    print('updated', name)
