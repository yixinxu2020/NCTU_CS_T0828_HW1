import os
import random
import pandas as pd
from shutil import copy2
from collections import defaultdict

# read the csv file with id, labels saved
df = pd.read_csv("training_labels.csv")
# get the labels, id and file_name list
labels = df['label'].values.tolist()
id_ = df['id'].values.tolist()
file_name = [str(i).zfill(6) + '.jpg' for i in id_]
labels_list = list(set(labels))
labels_number = len(labels_list)
d = defaultdict(list)
# get the dic{}, k: label, v: index_list
for k, va in [(v, i) for i, v in enumerate(labels)]:
    d[k].append(va)


# make 'train' and 'val' file to save all_labels img
def mk_total_dir(data_path):
    dic = ['train', 'val']
    for i in range(0, 2):
        current_path = data_path + dic[i] + '/'
        # judge whether the current path exists,
        # fails to create if it does, and succeeds if it does not
        is_exists = os.path.exists(current_path)
        if not is_exists:
            os.makedirs(current_path)
            print('successful ' + dic[i])
        else:
            print('is existed')
    return


# make all_labels dic to save img
def mk_class_dir(dic_path):
    for i in range(0, labels_number):
        current_class_path = os.path.join(dic_path, labels_list[i])
        is_exists = os.path.exists(current_class_path)
        if not is_exists:
            os.makedirs(current_class_path)
            print('successful ' + labels_list[i])
        else:
            print('is existed')


# copy the img from source dic to new dic depend its label into different dic
# Split these imgs proportionally(9:1) into 'train' and 'val' dic
def divide_train_validation_test(source_path, train_path, validation_path):
    mk_class_dir(train_path)
    mk_class_dir(validation_path)

    for i in range(0, labels_number):
        index = d[labels_list[i]]
        random.shuffle(index)
        train_image_index = index[0:int(0.9 * len(index))]
        validation_image_index = index[int(0.9 * len(index)):]

        for m in train_image_index:
            origins_image_path = source_path + file_name[m]
            new_train_image_path = train_path + labels_list[i] + '/' + file_name[m]
            print('train:  ' + labels_list[i] + '****' + file_name[m])
            copy2(origins_image_path, new_train_image_path)
        for n in validation_image_index:
            origins_image_path = source_path + file_name[n]
            new_val_image_path = validation_path + labels_list[i] + '/' + file_name[n]
            copy2(origins_image_path, new_val_image_path)
            print('val:   ' + labels_list[i] + '****' + file_name[n])


def main():
    data_path = 'F:/jiaotong_cs_course/CS_109fall/CV_DL/hw/hw1/data/'
    source_path = 'F:/jiaotong_cs_course/CS_109fall/CV_DL/hw/hw1/training_data/training_data/'
    train_path = 'F:/jiaotong_cs_course/CS_109fall/CV_DL/hw/hw1/data/train/'
    validation_path = 'F:/jiaotong_cs_course/CS_109fall/CV_DL/hw/hw1/data/val/'

    mk_total_dir(data_path)
    divide_train_validation_test(source_path, train_path, validation_path)


if __name__ == '__main__':
    main()
