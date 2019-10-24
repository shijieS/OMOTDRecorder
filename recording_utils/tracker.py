#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the MIT license.
#   For a copy, see <https://opensource.org/licenses/MIT>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
import numpy as np
import cv2
from .draw_utils import get_random_color, cv_draw_one_box, cv_draw_8_points, putPrettyTextPos

class Node:
    def __init__(self, box, visibility, others):
        self.box = box
        self.visibility = visibility
        self.others = others
    def get_bc(self):
        return int((self.box[0] + self.box[2])/2), int(self.box[3])
    def get_h(self):
        return int(self.box[3] - self.box[1])

class Track:
    def __init__(self, id):
        self.id = int(id)
        self.nodes = []
        self.age = 0
        self.max_node = 15
    def update(self, id, box, visibility, others):
        if self.id != int(id):
            return
        self.nodes += [Node(box, visibility, others)]
        self.age = 0
        if len(self.nodes) > self.max_node:
            self.nodes = self.nodes[1:self.max_node]

        return self

    def draw(self, frame, max_age):
        if len(self.nodes) == 0 or self.age != 0:
            return frame
        color = get_random_color(self.id)
        for i in range(1, len(self.nodes)):
            # draw line
            cv2.line(frame, self.nodes[i-1].get_bc(), self.nodes[i].get_bc(), color, 2)
            # draw rectangles
            center = self.nodes[i-1].get_bc()
            radius = round(self.nodes[i-1].get_h() / 5.0 + 0.5)
            box = int(center[0] - radius), int(center[1] - radius), int(center[0] + radius), int(center[1] + radius)
            cv_draw_one_box(frame, box, color, alpha=0.9, with_border=False)

        return frame


class SimplerTracker:
    def __init__(self):
        self.max_age = 30
        self.tracks = []

    def get_track_ids(self):
        return [t.id for t in self.tracks]

    def update(self, datas):
        self.datas = datas

        for t in self.tracks:
            t.age += 1

        if len(self.tracks) == 0:
            self.tracks = [Track(d[0]).update(d[0], d[1:5], d[5], d[6:]) for d in datas]
        else:
            for t in self.tracks:
                [t.update(d[0], d[1:5], d[5], d[6:]) for d in datas]
            track_ids = datas[:, 0].astype(int)
            old_track_ids = self.get_track_ids()
            new_track_index = [i for i, id in enumerate(track_ids) if id not in old_track_ids]
            self.tracks += [Track(datas[i, 0]).update(datas[i, 0], datas[i, 1:5], datas[i, 5], datas[i, 6:]) for i in new_track_index]

        self.tracks = [t for t in self.tracks if t.age < self.max_age]


    def draw(self, frame,
             with_track = False,
             with_8_points = False,
             with_boxes = False,
             with_vis = False,
             with_weather=False):
        if self.datas is None:
            return

        # draw tracks
        if with_track:
            for t in self.tracks:
                t.draw(frame, self.max_age)

        # draw 8 points
        if with_8_points:
            for d in self.datas:
                cv_draw_8_points(frame,
                                 datas=d[6:22])
        # draw boxes
        if with_boxes:
            for d in self.datas:
                text = ""
                if with_vis:
                    text = (text+"Vis:{:0.2f}").format(d[5]*100)
                frame = cv_draw_one_box(frame,
                                box=d[1:5],
                                color=get_random_color(d[0]),
                                text=text)
        # draw weather
        if with_weather:
            frame = putPrettyTextPos(frame,
                                     ("cloudyness: {:0.2f}; " +
                                      "precipitation: {:0.2f}; " +
                                      "sun altitude angle: {:0.2f}; " +
                                      "wind intensity {:0.2f}; ").format(
                                          self.datas[0, 25],
                                          self.datas[0, 26],
                                          self.datas[0, 27],
                                          self.datas[0, 28]
                                      ),
                                     (0, 0, 255),
                                     (0, 0, 0),
                                     pos=(0, 0),
                                     f = 0.7
                                  )

        return frame