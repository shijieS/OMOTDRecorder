#  Copyright (c) 2019. ShiJie Sun at the Chang'an University
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#  Author: shijie Sun
#  Email: shijieSun@chd.edu.cn
#  Github: www.github.com/shijieS

import numpy as np



class SVehicle():
    @staticmethod
    def get_rects(bounding_boxes):
        return [SVehicle.get_rect(bbx) for bbx in bounding_boxes]

    @staticmethod
    def get_rect(bounding_box):
        max_mat = np.max(bounding_box, axis=0)[0]
        min_mat = np.min(bounding_box, axis=0)[0]
        rect = np.array(np.vstack([min_mat, max_mat]))
        return rect