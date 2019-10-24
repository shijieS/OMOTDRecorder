#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the MIT license.
#   For a copy, see <https://opensource.org/licenses/MIT>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS

import cv2
import os
import glob
from tqdm import trange
import argparse
argparser = argparse.ArgumentParser(description='Convert Images to Video')
argparser.add_argument('--root_path', default='/home/ssj/Data/github/AwesomeMOTDataset/Dataset/Town05/All/Clear/100', help='image root path which contains gt and img')

args = argparser.parse_args()
# root_path = "/home/ssj/Data/github/AwesomeMOTDataset/Dataset/Town05/All/Clear/100"


def start_convert(root_path):
    image_root_path = os.path.join(root_path, 'img')
    image_path_list = glob.glob(os.path.join(image_root_path, '*'))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    for _, image_path in zip(trange(len(image_path_list)), image_path_list):
        base_name = os.path.basename(image_path)
        video_path = os.path.join(root_path, "{}.avi".format(base_name))
        vw = cv2.VideoWriter(video_path, fourcc, 30, (1920, 1080))
        for i in trange(3000):
            img = cv2.imread(os.path.join(image_path, "{}.jpg".format(i)))
            if img is None:
                raise("Epty frame!!!!!")
            vw.write(img)
            cv2.waitKey(int(1000/25))


if __name__ == "__main__":
    start_convert(args.root_path)