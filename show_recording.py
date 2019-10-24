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
from recording_utils import VideoCaptureAsync, cv_draw_mult_boxes, SimplerTracker
import tqdm

argparser = argparse.ArgumentParser(
        description='Awesome showing recording data script')

argparser.add_argument(
        '--root_folder',
        default='/media/ssj/新加卷/temp/h-3',
        help='A root folder contains the gt folder and img folder')

argparser.add_argument(
        '--is_video',
        default=True,
        help='reading from video or from image folders')

args = argparser.parse_args()

class SingleRecordingData:
    def __init__(self, csv_file_path, file, base_name, min_integrity, is_video=False):
        self.is_video = is_video
        self.csv_file_path = csv_file_path
        self.label_data = pd.read_csv(csv_file_path, index_col=False)
        self.bboxes = self.label_data.loc[:, ['frame_idx', 'id', 'l', 't', 'r', 'b', 'integrity',
                                              'pt0_x', 'pt0_y', 'pt1_x', 'pt1_y',
                                              'pt2_x', 'pt2_y', 'pt3_x', 'pt3_y',
                                              'pt4_x', 'pt4_y', 'pt5_x', 'pt5_y',
                                              'pt6_x', 'pt6_y', 'pt7_x', 'pt7_y',
                                              'velocity_x', 'velocity_y', 'velocity_z',
                                              'cloudyness', 'precipitation', 'sun_altitude_angle',
                                              'wind_intensity']]
        self.bboxes_group = self.bboxes.groupby(self.bboxes['frame_idx'])
        self.data = {int(k):v.iloc[:, 1:].to_numpy() for k, v in self.bboxes_group}

        if self.is_video:
            self.video_file = file
        else:
            self.image_folder = file
            self.image_path_format = os.path.join(self.image_folder, "{}.jpg")
            self.image_files = [self.image_path_format.format(int(k)) for k in self.bboxes_group.keys()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if item not in self.data.keys():
            return None, None

        if self.is_video:
            ret, frame = VideoCaptureAsync.get_frame(self.video_file, item)
            return frame, self.data[item]
        else:
            return cv2.imread(self.image_files[item]), self.data[item]


class RecordingData:
    def __init__(self, root_folder, is_video=False):
        self.is_video = is_video
        self.gt_folder = os.path.join(root_folder, 'gt')

        self.csv_paths = glob.glob(os.path.join(self.gt_folder, "*.csv"))
        self.base_names = [os.path.splitext(os.path.basename(p))[0] for p in self.csv_paths]

        print("Reading data....")
        self.datas = []
        if is_video:
            self.video_folder = root_folder
            self.video_paths = [os.path.join(self.video_folder, b + ".avi") for b in self.base_names]
            for csv_path, video_path, base_name in tqdm.tqdm(zip(self.csv_paths, self.video_paths, self.base_names)):
                if not os.path.exists(video_path) or not os.path.exists(csv_path):
                    continue
                self.datas += [SingleRecordingData(csv_path, video_path, base_name, min_integrity=0.3, is_video=self.is_video)]
        else:
            self.image_folder = os.path.join(root_folder, 'img')
            self.image_folders = [os.path.join(self.image_folder, b) for b in self.base_names]
            for csv_path, image_folder, base_name in zip(self.csv_paths, self.image_folders, self.base_names):
                if not os.path.exists(image_folder) or not os.path.exists(self.csv_paths):
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


if __name__ == "__main__":
    rd = RecordingData(args.root_folder, args.is_video)
    tracker = SimplerTracker()

    with_track = False,
    with_8_points = False,
    with_boxes = False,
    with_vis = False
    with_weather=False

    for frame, bboxes in rd:
        if frame is None or bboxes is None:
            continue
        tracker.update(bboxes)
        frame = tracker.draw(frame, with_track, with_8_points, with_boxes, with_vis, with_weather)
        cv2.imshow("result", frame)
        c = cv2.waitKey(0)
        if c == ord('t'):
            with_track = not with_track
        elif c == ord('b'):
            with_boxes = not with_boxes
        elif c == ord('v'):
            with_vis = not with_vis
        elif c == ord('p'):
            with_8_points = not with_8_points
        elif c == ord('w'):
            with_weather = not with_weather
