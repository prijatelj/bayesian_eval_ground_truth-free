import numpy as np
import os
import imageio
import pandas as pd
import cv2
import ast
import matplotlib.pyplot as plt

from psych_metric.datasets.base_dataset import BaseDataset

ROOT = os.environ['ROOT']
HERE = os.path.join(ROOT, 'data/datasets/aflw/')

class AFLW(BaseDataset):
    def __init__(self, stage='train', augment=False, numpy=False
        self.numpy = numpy
        self.augment = augment
        self.stage = stage
        self.annotations = os.path.join(HERE, 'annotations_att.csv')
        if self.numpy:
            self.img_dir = os.path.join(HERE, 'aflw-att-numpy')
        else:
            self.img_dir = os.path.join(HERE, 'aflw-att-images')
        self.df = self.load_csv()

    def jpg_to_numpy(self, f):
        return f[:-len('jpg')] + 'npy'

    def load_csv(self):
        df = pd.read_csv(self.annotations, index_col=0, header=0)
        df = df[df['file_id'].map(lambda x: isinstance(x, str))]
        df = df[
            (df['Trustworthiness_num'] != 0) & \
            (df['Dominance_num'] != 0) & \
            (df['Age_num'] != 0) & \
            (df['IQ_num'] != 0)
        ]
        if self.numpy:
            df['file_id'] = df['file_id'].map(self.jpg_to_numpy)
        df = df[df['split'] == self.stage]

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        i = i % len(self.df)
        row = self.df.iloc[i]
        imf = os.path.join(self.img_dir, row['file_id'])

        if self.numpy:
            img = np.load(imf)
        else:
            img = imageio.imread(imf)

        # cut out the image with jitter at train time
        box = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        if self.augment: box = utils.jitter_box(box, pw=0.2, ph=0.2)
        box = utils.make_box_square(box)
        roi = utils.cut_out_box(img, box, pad_mode='edge')
        roi = cv2.resize(roi, (224,224))

        # return trustworthiness, dominance, age, iq
        age = row['Age_raw']
        trust = row['Trustworthiness_raw']
        dom = row['Dominance_raw']
        iq = row['IQ_raw']

        raw_to_mean = lambda x: (np.mean(ast.literal_eval(x)) - 1.) / 6.0
        for arr in [age, trust, dom, iq]:
            arr = ast.literal_eval(arr)
            if len(arr) == 0:
                print(i)

        ret = {
            'Trustworthiness': raw_to_mean(trust),
            'Dominance': raw_to_mean(dom),
            'Age': raw_to_mean(age),
            'IQ': raw_to_mean(iq),
            'image': roi,
        }

        return ret

    def dislay(self, i, ax=None):
        r = self[i]
        if ax is None: fig, ax = plt.subplots(1,1 figsize=(10,10))
        title = [k + "{:.2f}".format(v) for k, v in r.items() if k != 'image'].join('\n')
        ax.imshow(r['image'])
        ax.set_title(title)