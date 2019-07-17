# --------------------------------------------------------
# updated by wang lei
# update time 2018-10-9 11:02:56
# --------------------------------------------------------

import xml.etree.ElementTree as ET

long_name = ('cement pole broken', 'cement pole leaks steel',
             'cement pole top is leaky', 'pdz lost', 'bird nest')
short_name = ('poleBroken', 'poleLeaksSteel',
              'poleTopLeaky', 'pdzLost', 'nest')


def format_xml(elem, level=0):
    """
    format xml file
    :param elem:
    :param level:
    :return:
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            format_xml(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def build_xml(obj_list, img_name, img_path, img_shape, save_path):
    """
    create xml file
    :param obj_list:
    :param img_name:
    :param img_path:
    :param img_shape:
    :param save_path:
    :return:
    """
    node_root = ET.Element('annotation')

    node_folder = ET.SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = ET.SubElement(node_root, 'filename')
    node_filename.text = img_name + '.jpg'

    node_path = ET.SubElement(node_root, 'path')
    node_path.text = img_path

    node_source = ET.SubElement(node_root, 'source')
    source_database = ET.SubElement(node_source, 'database')
    source_database.text = 'Unknown'

    node_size = ET.SubElement(node_root, 'size')
    size_width = ET.SubElement(node_size, 'width')
    size_width.text = str(img_shape[1])

    size_height = ET.SubElement(node_size, 'height')
    size_height.text = str(img_shape[0])

    size_depth = ET.SubElement(node_size, 'depth')
    size_depth.text = str(img_shape[2])

    for p in obj_list:
        node_object = ET.SubElement(node_root, 'object')

        obj_name = ET.SubElement(node_object, 'name')
        obj_name.text = p[0]

        obj_pose = ET.SubElement(node_object, 'pose')
        obj_pose.text = p[1]

        obj_truncated = ET.SubElement(node_object, 'truncated')
        obj_truncated.text = p[2]

        obj_difficult = ET.SubElement(node_object, 'difficult')
        obj_difficult.text = p[3]

        obj_bndbox = ET.SubElement(node_object, 'bndbox')

        bndbox_xmin = ET.SubElement(obj_bndbox, 'xmin')
        bndbox_xmin.text = p[4]
        bndbox_ymin = ET.SubElement(obj_bndbox, 'ymin')
        bndbox_ymin.text = p[5]
        bndbox_xmax = ET.SubElement(obj_bndbox, 'xmax')
        bndbox_xmax.text = p[6]
        bndbox_ymax = ET.SubElement(obj_bndbox, 'ymax')
        bndbox_ymax.text = p[7]

    tree = ET.ElementTree(node_root)
    format_xml(node_root)
    tree.write(save_path, encoding="UTF-8")


def parse_xml(xml_file, chip_shape):
    """
    parse xml file get object
    :param xml_file:
    :param chip_shape:
    :return:
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    obj_list = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        pose = obj.find('pose').text
        truncated = obj.find('truncated').text
        difficult = obj.find('difficult').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        if out_of_image(xmin, ymin, xmax, ymax, chip_shape):
            continue
        xmin = str(int(xmin) - chip_shape[0])
        ymin = str(int(ymin) - chip_shape[1])
        xmax = str(int(xmax) - chip_shape[0])
        ymax = str(int(ymax) - chip_shape[1])
        data = (name, pose, truncated, difficult, xmin, ymin, xmax, ymax)
        obj_list.append(data)
    return obj_list


def update_obj_name(xml_file):
    """
    update object name
    :param xml_file: xml file path
    :return: save as the same file name after update
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name not in long_name:
            continue
        else:
            index = long_name.index(name)
            obj.find('name').text = short_name[index]
    tree.write(xml_file, encoding="UTF-8")


def out_of_image(xmin, ymin, xmax, ymax, img_shape):
    """
    judge bounding box is out of image
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :param img_shape:
    :return:
    """
    if len(img_shape) == 0:
        return True
    x0 = img_shape[0] <= int(xmin)
    y0 = img_shape[1] <= int(ymin)
    x1 = img_shape[2] >= int(xmax)
    y1 = img_shape[3] >= int(ymax)
    if x0 and y0 and x1 and y1:
        return False
    else:
        return True


if __name__ == "__main__":
    update_obj_name('D:/MyWorks/python/MyRFBNet/utils/000118.xml')
