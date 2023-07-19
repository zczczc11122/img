import random
from torchvision import transforms
import numpy as np
from PIL import Image

# class DarkTran(object):
#     def __init__(self):
#         self.trans = transforms.ColorJitter(brightness=(0.3, 0.4), contrast=(1, 1.2), saturation=(1, 1.5))
#         self.p = 0.5
#     def __call__(self, img):
#         prob = random.uniform(0, 1)
#         if prob < self.p:
#             img = self.trans(img)
#         return img

class DarkTran(object):
    def __init__(self):
        self.p = 0.5
    def __call__(self, img):
        prob = random.uniform(0, 1)
        if prob < self.p and float(np.max(img))>0:
            img = np.array(img)
            img = np.power(img / float(np.max(img)), 4)
            img = np.array(img * 255, np.uint8)
            img = Image.fromarray(img)
        return img