import os
import re
import xml.etree.ElementTree as ET


def get_image_info(root_path):
    annotation_root = ET.parse(root_path)
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(root_path, label2id):
    annotation_root = ET.parse(root_path)
    objects = annotation_root.findall('object')
    anns = []
    for obj in objects:
        label = obj.findtext('name')
        assert label in label2id, f"Error: {label} is not in label2id !"
        category_id = label2id[label]
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext('xmin')) - 1
        ymin = int(bndbox.findtext('ymin')) - 1
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))
        assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        o_width = xmax - xmin
        o_height = ymax - ymin
        ann = {
            'bbox': [xmin, ymin, o_width, o_height],
            'category_id': category_id,
            }
        anns.append(ann)
    return anns
