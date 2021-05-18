import os
import shutil
import json
import cv2
import tqdm

BASE_IMG_PATH = '.\\df2\\val\\image'
BASE_JSON_PATH = '.\\df2\\val\\annos'
BASE_OUT_PATH = '.\\df2_parsed\\val'

for json_filename in tqdm.tqdm(os.listdir(os.path.join(BASE_JSON_PATH))):
    json_path = os.path.join(BASE_JSON_PATH, json_filename)
    filename = json_filename.split('.')[0]
    file_path = os.path.join(BASE_IMG_PATH, filename) + '.jpg'

    with open(json_path, 'r') as f:
        im = cv2.imread(file_path)
        json_raw = f.read()
        json_obj = json.loads(json_raw)
        json_item_keys = list(filter(lambda x : (x.startswith('item')), json_obj.keys()))
        for item_key in json_item_keys:
            x1, y1, x2, y2 = json_obj[item_key]['bounding_box']
            category = json_obj[item_key]['category_name']
            target_filedir = os.path.join(BASE_OUT_PATH, category)
            if not os.path.isdir(target_filedir):
                os.makedirs(target_filedir)
            im_cropped = im[y1:y2, x1:x2]
            target_filename = filename + "_" + item_key + '.jpg'
            target_filepath = os.path.join(target_filedir, target_filename)
            cv2.imwrite(target_filepath, im_cropped)


