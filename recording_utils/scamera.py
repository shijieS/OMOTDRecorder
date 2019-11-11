#  Copyright (c) 2019. ShiJie Sun at the Chang'an University
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#  Author: shijie Sun
#  Email: shijieSun@chd.edu.cn
#  Github: www.github.com/shijieS

import carla
import numpy as np
import queue
import pygame
import cv2
import json
from .svehicle import SVehicle
from collections import OrderedDict
import pandas as pd


class SCamera:
    def __init__(self, camera, camera_depth):
        self.camera = camera
        self.camera_depth = camera_depth
        self.update_intrinsic_matrix()
        self.queue = None
        self.queue_depth = None
        self.display = None
        self.gt_data = pd.DataFrame(columns=["frame_idx", "id",     # frame index and the object's id
                                             "l", 't', 'r', 'b',    # object rectangles
                                             "pt0_x", "pt0_y",      # 8 corner of the object bbox
                                             "pt1_x", "pt1_y",
                                             "pt2_x", "pt2_y",
                                             "pt3_x", "pt3_y",
                                             "pt4_x", "pt4_y",
                                             "pt5_x", "pt5_y",
                                             "pt6_x", "pt6_y",
                                             "pt7_x", "pt7_y",
                                             "physical_x", "physical_y", "physical_z",
                                             "integrity",  # the integrity ratio
                                             "velocity_x", "velocity_y", "velocity_z",              # velocity
                                             "acceleration_x", "acceleration_y", "acceleration_z",  # acceleration
                                             "number_of_wheels",
                                             "camera_w", "camera_h", "camera_fov",  # camera parameters
                                             "camera_x", "camera_y", "camera_z",
                                             "camera_pitch", "camera_yaw", "camera_roll",
                                             "cloudyness", "precipitation", "sun_altitude_angle", "wind_intensity"]   # weather
                                    )
        self.insert_camera_index = 0

    def get_bounding_boxes(self, vehicles):
        """
        Get the vehicles' 3D bounding boxes projected onto the image plane, object rectangles and the vehicles' id
        :param vehicles: all the vehicles.
        :return: 3D bounding boxes, object rectangles, and each vehicle's id
        """
        bounding_boxes = [self.get_bounding_box(vehicle) for vehicle in vehicles]
        # filter objects behind camera
        # filter objects behind camera
        indexes = [i for i, bb in enumerate(bounding_boxes) if all(bb[:, 2] > 0)]

        bounding_boxes = [bounding_boxes[i] for i in indexes]
        rects = SVehicle.get_rects(bounding_boxes)
        ids = [vehicles[i].id for i in indexes]
        physical_params = [self.get_physical_params(vehicles[i]) for i in indexes]
        return bounding_boxes, rects, ids, physical_params, indexes

    def get_physical_params(self, vehicle):
        extent = vehicle.bounding_box.extent
        return (extent.x*2, extent.y*2, extent.z*2)

    def get_bounding_box(self, vehicle):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = self._create_bb_points(vehicle)
        cords_x_y_z = self._vehicle_to_sensor(bb_cords, vehicle)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(self.intrinsic_matrix, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    def update_intrinsic_matrix(self):
        self.width = int(self.camera.attributes['image_size_x'])
        self.height = int(self.camera.attributes['image_size_y'])
        self.fov = float(self.camera.attributes['fov'])

        self.intrinsic_matrix = np.identity(3)
        self.intrinsic_matrix[0, 2] = self.width / 2.0
        self.intrinsic_matrix[1, 2] = self.height / 2.0
        self.intrinsic_matrix[0, 0] = self.intrinsic_matrix[1, 1] = self.width / (2.0 * np.tan(self.fov * np.pi / 360.0))

    def _vehicle_to_sensor(self, cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = SCamera._vehicle_to_world(cords, vehicle)
        sensor_cord = self._world_to_sensor(world_cord)
        return sensor_cord

    def _world_to_sensor(self, cords):
        """
        Transforms world coordinates to sensor.
        """
        sensor_world_matrix = SCamera.get_matrix(self.camera.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    def _sensor_to_world(self, cords):
        sensor_world_matrix = SCamera.get_matrix(self.camera.get_transform())
        world_cords = np.dot(sensor_world_matrix, cords)
        return world_cords

    def _apply_transform(self, x, y, z, pitch, roll, yaw):
        t = self.camera.get_transform()
        l = t.location
        r = t.rotation
        world_vec = self._sensor_to_world(np.array([[x, y, z, 1]]).T)

        self.camera.set_transform(carla.Transform(
            carla.Location(world_vec[0, 0], world_vec[1, 0], world_vec[2, 0]),
            carla.Rotation(pitch=r.pitch+pitch*10,
                           roll=r.roll+roll*10,
                           yaw=r.yaw+yaw*10)
        ))


    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = SCamera.get_matrix(bb_transform)
        vehicle_world_matrix = SCamera.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def add_listen_queue(self):
        """
        Create and add the listening queue to the camera
        """
        self.queue = queue.Queue()
        self.queue_depth = queue.Queue()

        self.camera.listen(self.queue.put)
        self.camera_depth.listen(self.queue_depth.put)

    def get_image(self, gt_frame_num=None):
        """
        Get image from the listening queue. If gt_frame_num is not None, then this will find this frame and return it.
        :param gt_frame_num: ground truth frame.
        :return: frame in the queue.
        """
        if self.queue is None:
            return None
        if gt_frame_num is None:
            return self.queue.get()
        else:
            while True:
                image = self.queue.get()
                if image.frame_number == gt_frame_num:
                    break
            return image

    def reset_depth_camera(self):
        self.camera_depth.set_transform(
            self.camera.get_transform()
        )

    def get_image_depth(self, gt_frame_num=None):
        """
        Get depth image from depth queue. If gt_frame_num is not None, then get the depth image at specified frame index.
        :param gt_frame_num: the specified depth frame.
        :return: depth image
        """

        if self.queue_depth is None:
            return None
        if gt_frame_num is None:
            return self.queue_depth.get()
        else:
            while True:
                image = self.queue_depth.get()
                if image.frame_number == gt_frame_num:
                    break
            return image

    @staticmethod
    def depth_to_array(image):
        """
        Convert an image containing CARLA encoded depth-map to a 2D array containing
        the depth value of each pixel normalized it in meters.
        """
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)

        return 1000*normalized_depth


    def get_overlap_ratio(self, rects, depth_image):
        """
        Get the overlap ratio of each detected boxes.

        We use the center box of rectangle and compare with the corresponding patch in the depth image.
        After that, we can get the z value of this patch, then cacluate the ratio of pixel between min z and max z.
        :param rects: all the rectangles
        :param depth_image: the depth image
        :return: overlap ratio list
        """
        array = SCamera.depth_to_array(depth_image)

        ratios = -np.ones(len(rects))
        # remove rects whose center is out of image

        for i, r in enumerate(rects):
            # remove center is not in the image
            center = (int(round((r[0, 0] + r[1, 0]) / 2.0)), int(round((r[0, 1] + r[1, 1]) / 2.0)))
            if center[0] < 0 or center[0] >= self.width or center[1] < 0 or center[1] >= self.height:
                continue

            center_block = np.meshgrid(
                range(int(round(r[0, 0])), int(round(r[1, 0]))),
                range(int(round(r[0, 1])), int(round(r[1, 1]))))

            h = (r[1, 1] - r[0, 1])
            w = (r[1, 0] - r[0, 0])
            all_z_values = array[
                           int(round(r[0, 1]+h/4)):int(round(r[1, 1]-h/4)),
                           int(round(r[0, 0]+w/4)) : int(round(r[1, 0]-w/4))
                           ]
            # keep center between the max and min value
            r = np.mean(np.logical_and(all_z_values > r[0, 2], all_z_values < r[1, 2]))
            if r < 1e-3:
                continue

            ratios[i] = r
        return ratios

    def add_display(self):
        if self.display is None:
            self.display = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    def display_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        self.display.blit(image_surface, (0, 0))

    def display_cv_image(self, image):
        """
        Display opencv format BGR image
        :param image: opencv formatted image
        :return: None
        """
        array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.display.blit(image_surface, (0, 0))

    # Camera Control Part
    control_dict = {
        'w': np.array([0.1, 0, 0]),
        's': np.array([-0.1, 0, 0]),
        'a': np.array([0, -0.1, 0]),
        'd': np.array([0, 0.1, 0]),
        'q': np.array([0, 0, 0.1]),
        'e': np.array([0, 0, -0.1]),

        'i': np.array([0.1, 0, 0]),
        'k': np.array([-0.1, 0, 0]),
        'j': np.array([0, 0, -0.1]),
        'l': np.array([0, 0, 0.1]),
        'u': np.array([0, -0.1, 0]),
        'o': np.array([0, 0.1, 0])
    }
    def on_key_w_s_a_d(self, key, scale=30):
        # Control the camera
        if key in ['w', 's', 'a', 'd', 'q', 'e']:
            self._apply_transform(*tuple(SCamera.control_dict[key]*scale), 0, 0, 0)
        else:
            self._apply_transform(0, 0, 0, *SCamera.control_dict[key])

    def save_transform(self, save_path, level_str):
        t = self.camera.get_transform()
        l = t.location
        r = t.rotation

        param = OrderedDict({
            level_str+"_Camera_{}".format(self.insert_camera_index): self.get_camera_params()})
        self.insert_camera_index += 1

        with open(save_path, 'a+') as f:
            js = json.dump(param, f)
            f.write("\n")

    def get_camera_params(self):
        t = self.camera.get_transform()
        l = t.location
        r = t.rotation
        param = dict([
            ('width', self.width),
            ('height', self.height),
            ('x', l.x),
            ('y', l.y),
            ('z', l.z),
            ('fov', float(self.camera.attributes['fov'])),
            ('roll', r.roll),
            ('yaw', r.yaw),
            ('pitch', r.pitch),
            ('max_record_frame', 10000)
        ])
        return param

    def update_gt_data(self, frame_idx, id, rect, bbox, ratio, physical_param, vehicle):
        """
        Update current ground truth data
        :param frame_idx: frame index
        :param id: object id
        :param rect: object rectangle
        :param bbox: object 8 3D bounding box
        :param ratio: overlapped ratio
        :return: None
        """
        t = self.camera.get_transform()
        l = t.location
        r = t.rotation
        w = self.camera.get_world().get_weather()
        velocity = vehicle.get_angular_velocity()
        acceleration = vehicle.get_acceleration()
        number_of_wheels = int(vehicle.attributes["number_of_wheels"])
        self.gt_data = self.gt_data.append(pd.Series([
            frame_idx, id,
            rect[0, 0], rect[0, 1], rect[1, 0], rect[1, 1],
            *tuple([bbox[i//2, i%2] for i in range(16)]),
            *physical_param,
            ratio,
            velocity.x, velocity.y, velocity.z,
            acceleration.x, acceleration.y, acceleration.z,
            number_of_wheels,
            self.width, self.height, self.fov,
            l.x, l.y, l.z,
            r.pitch, r.roll, r.yaw,
            w.cloudyness, w.precipitation, w.sun_altitude_angle, w.wind_intensity
        ], index=self.gt_data.columns), ignore_index=True)

    def save_gt_data(self, file_path):
        self.gt_data.to_csv(file_path, index=False)

    def stop(self):
        self.camera.stop()
        self.camera_depth.stop()
        pygame.display.quit()