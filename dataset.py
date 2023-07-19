import os
import torch.utils.data as data
import pandas as pd
from PIL import Image
import logging

logger = logging.getLogger('vit')


class JumpRecord(object):
    def __init__(self, path, label):
        self._path = path
        self._label = label

    @property
    def path(self):
        return self._path

    @property
    def label(self):
        return int(self._label)


class DataSet_Jump(data.Dataset):
    def __init__(self, file_path, mode, pre_path_v1, data_transform):
        '''
        :param file_path:
        :param mode: train test dev
        :param pre_path:
        :param label_mode: info human
        '''
        self.transform = data_transform
        self.img_list = []
        exc = pd.read_excel(file_path, dtype=object)
        lines = exc.values.tolist()
        for i in range(len(lines)):
            lines[i] = lines[i][:3] + [pre_path_v1]
        node = int(len(lines) * 0.8)
        if mode == "train":
            lines_sub = lines[:node]
            lines = lines_sub
        else:
            lines = lines[node:]

        record = ''
        for i in lines:
            vid, img_id, label, pre_path = i
            if pd.isna(label):
                label = 1
            if pd.isna(vid):
                vid = record
            else:
                record = vid
            if record == '':
                continue
            if pd.isna(img_id):
                logger.info("{} {} {}".format(vid, img_id, label))
                continue
            if int(label) not in [0, 1]:
                continue
            if not os.path.exists(os.path.join(pre_path, vid, img_id)):
                logger.info("{} not exists".format(os.path.join(pre_path, vid, img_id)))
                continue
            self.img_list.append(JumpRecord(os.path.join(pre_path, vid, img_id), int(label)))

    def __getitem__(self, index):
        record = self.img_list[index]
        img, label = self._get(record)
        img = self.transform(img)
        return img, label, record.path

    def __len__(self):
        return len(self.img_list)

    def _get(self, record):
        img_path = record.path
        label = record.label
        img = self._load_images(img_path)
        return img, label

    def _load_images(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image