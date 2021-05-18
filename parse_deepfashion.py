import shutil
import os
import tqdm

BASE_IMAGE_PATH = '.\\img'
BASE_OUTPUT_PATH = '.\\img_parsed'
TEST_LIST = '.\\test.txt'
TEST_CATE_LIST = '.\\test_cate.txt'
VAL_LIST = '.\\val.txt'
VAL_CATE_LIST = '.\\val_cate.txt'
TRAIN_LIST = '.\\train.txt'
TRAIN_CATE_LIST = '.\\train_cate.txt'


def get_list_from_file(list_filepath, cate_filepath):
    with open(list_filepath, 'r') as f:
        file_list = f.read().splitlines()
    with open(cate_filepath, 'r') as f:
        cate_list = f.read().splitlines()
    return file_list, cate_list

def make_dirs(file_list, cate_list, folder_name):
    target_path = os.path.join(BASE_OUTPUT_PATH, folder_name)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    else:
        return

    for a in tqdm.tqdm(range(len(file_list))):
        category = cate_list[a]
        source_path = file_list[a]
        source_filename_split = source_path.split('/')
        file_name = source_filename_split[-2] + "_" + source_filename_split[-1]

        final_path = os.path.join(target_path, category)
        if not os.path.isdir(final_path):
            os.makedirs(final_path)

        final_filepath = os.path.join(final_path, file_name)
        shutil.copy(source_path, final_filepath)

if __name__ == '__main__':
    train_list, train_cate_list = get_list_from_file(TRAIN_LIST, TRAIN_CATE_LIST)
    val_list, val_cate_list = get_list_from_file(VAL_LIST, VAL_CATE_LIST)
    test_list, test_cate_list = get_list_from_file(TEST_LIST, TEST_CATE_LIST)
    make_dirs(train_list, train_cate_list, 'train')
    make_dirs(val_list, val_cate_list, 'val')
    make_dirs(test_list, test_cate_list, 'test')





