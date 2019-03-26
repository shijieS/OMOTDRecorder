import carla
import numpy as np
import queue
import pygame
import cv2


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