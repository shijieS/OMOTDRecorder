#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the MIT license.
#   For a copy, see <https://opensource.org/licenses/MIT>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS

import argparse
import pandas as pd
import os
import glob
import numpy as np
import cv2

argparser = argparse.ArgumentParser(
        description='Awesome showing recording data script')

argparser.add_argument(
        '--root_folder',
        default='/home/ssj/Data/github/AwesomeMOTDataset/Dataset/Town04/Easy/Clear/100',
        help='A root folder contains the gt folder and img folder')

args = argparser.parse_args()

class SingleRecordingData:
    def __init__(self, csv_file_path, image_folder, base_name, min_integrity):
        self.csv_file_path = csv_file_path
        self.image_folder = image_folder
        self.label_data = pd.read_csv(csv_file_path, index_col=False)

        self.bboxes = self.label_data.loc[:, ['frame_idx', 'id', 'l', 't', 'r', 'b', 'integrity']]
        self.bboxes_group = self.bboxes.groupby(self.bboxes['frame_idx'])
        self.image_path_format = os.path.join(image_folder, "{}.jpg")

        self.data = {}
        for k, v in self.bboxes_group:
            self.data[int(k)] = (v.iloc[:, 1:].to_numpy(), self.image_path_format.format(int(k)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if item not in self.data.keys():
            return None, None

        data = self.data[item]
        return cv2.imread(data[1]), data[0]


class RecordingData:
    def __init__(self, root_folder):
        self.gt_folder = os.path.join(root_folder, 'gt')
        self.image_folder = os.path.join(root_folder, 'img')

        self.csv_paths = glob.glob(os.path.join(self.gt_folder, "*.csv"))
        self.base_names = [os.path.splitext(os.path.basename(p))[0] for p in self.csv_paths]
        self.image_folders = [os.path.join(self.image_folder, b) for b in self.base_names]

        self.datas = []

        for csv_path, image_folder, base_name in zip(self.csv_paths, self.image_folders, self.base_names):
            if not os.path.exists(image_folder) or not os.path.exists(csv_path):
                continue

            self.datas += [SingleRecordingData(csv_path, image_folder, base_name, min_integrity=0.3)]

        self.lens = [len(d) for d in self.datas]

    def __len__(self):
        return np.sum(self.lens)

    def __getitem__(self, item):
        i=0
        rest_len = item
        while True:
            if rest_len - self.lens[i] < 0:
                break
            i += 1
            rest_len -= self.lens[i]

        return self.datas[i][rest_len]




def cv_draw_one_box(frame,
                    box,
                    color,
                    content_color=None,
                    alpha=0.5,
                    text="",
                    font_color=None,
                    with_border=True,
                    border_color=None):
    """
    Draw a box on a frame
    :param frame:
    :param box:
    :param color:
    :param alpha:
    :param text:
    :param font_color:
    :param with_border:
    :param border_color:
    :return:
    """
    # draw box content
    if content_color is None:
        content_color = color
    (l, t, r, b) = tuple([int(b) for b in box])
    roi = frame[t:b, l:r]
    black_box = np.zeros_like(roi)
    black_box[:, :, :] = content_color
    cv2.addWeighted(roi, alpha, black_box, 1-alpha, 0, roi)
    # draw border
    if with_border:
        if border_color is None:
            border_color = color
        cv2.rectangle(frame, (l, t), (r, b), border_color, 1)
    # put text
    if font_color is None:
        font_color = color
    cv2.putText(frame, text, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color)
    return frame


def cv_draw_mult_boxes(frame, boxes, colors=None):
    """
    Draw multiple boxes on one frame
    :param frame: the frame to be drawn
    :param boxes: all the boxes, whoes shape is [n, 4]
    :param color: current boxes' color
    :return:
    """
    boxes_len = len(boxes)
    if colors is None:
        colors = [get_random_color(i) for i in range(boxes_len)]
    for box, color in zip(boxes, colors):
        frame = cv_draw_one_box(frame, box, color)
    return frame


def get_random_color(seed=None):
    """
    Get the random color.
    :param seed: if seed is not None, then seed the random value
    :return:
    """
    if seed is not None:
        np.random.seed(seed)
    return tuple([np.random.randint(0, 255) for i in range(3)])

def get_random_colors(num, is_seed=True):
    """
    Get a set of random color
    :param num: the number of color
    :param is_seed: is the random seeded
    :return: a list of colors, i.e. [(255, 0, 0), (0, 255, 0)]
    """
    if is_seed:
        colors = [get_random_color(i) for i in range(num)]
    else:
        colors = [get_random_color() for _ in range(num)]
    return colors


if __name__ == "__main__":
    rd = RecordingData(args.root_folder)

    for frame, bboxes in rd:
        if frame is None or bboxes is None:
            continue

        cv_draw_mult_boxes(frame, bboxes[:, 1:-1])
        cv2.imshow("result", frame)
        cv2.waitKey(0)

