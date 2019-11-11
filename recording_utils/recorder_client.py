#  Copyright (c) 2019. ShiJie Sun at the Chang'an University
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#  Author: shijie Sun
#  Email: shijieSun@chd.edu.cn
#  Github: www.github.com/shijieS

import os
import carla
import pygame
import numpy as np
import cv2
try:
    import queue
except ImportError:
    import Queue as queue
from recording_utils import SCamera
import random
from collections import OrderedDict
import time
import weakref
import math
import sys
from tqdm import trange
import json


class SRecorder:
    def __init__(self, host='localhost', port=2000,
                 weather_name="Clear",
                 vehicle_num=100,
                 save_root='./recorders/',
                 flag_show_windows=False,
                 auto_save=True,
                 config=None,
                 config_file=None):
        """
        This recorder is used for recording the detected boxes in RGB camera
        :param host: host ip
        :param port: host port
        :param save_root: the default save root. Files would be saved in this folder.
        """
        self.host = host
        self.port = port
        self.save_root = save_root
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(100.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.actor_list = []
        self.vehicle_list = []
        self.camera_dict = OrderedDict()
        self.display_camera = 0
        self.current_camera = None

        self.flag_show_3D_bbox = False
        self.flag_show_rect = False
        self.flag_show_label = False
        self.flag_image_process = auto_save
        self.flag_show_windows = flag_show_windows

        self.config = config
        self.config_file = config_file

        # Set the yellow light longer considering the traffic jam
        all_traffic_ligts = self.world.get_actors().filter('*.traffic_light')
        for tl in all_traffic_ligts:
            tl.set_yellow_time(5)
            tl.set_green_time(7)

        for s in self.world.get_actors().filter('*.stop'):
            s.destroy()

        self.move_scale = 30
        self.camera_level_str = ['Easy', "Middle", "Hard"]
        self.current_select_level = 0
        self.log_display_str = ""

        # get 3 types of weather: Clear, Cloudy, Rain
        self.weather_conditions = {
            "Clear": [x for x in dir(carla.WeatherParameters) if 'Clear' in x],
            "Cloudy": [x for x in dir(carla.WeatherParameters) if 'Cloudy' in x],
            "Rain": [x for x in dir(carla.WeatherParameters) if 'Rain' in x]
        }

        self.weather_name = weather_name
        self.vehicle_num = vehicle_num
        print("start recording {} with {} vehicles spawned >>>>>".format(self.weather_name, self.vehicle_num))
        self.change_weather(self.weather_name)
        self.spaw_vehicles(self.vehicle_num)

    def random_get_weather(self, name):
        self.weather_name = name
        selection = list(range(1, len(self.weather_conditions[name])))
        np.random.shuffle(selection)
        weather_str = self.weather_conditions[name][selection[0]]
        return getattr(carla.WeatherParameters, weather_str)

    def _get_all_vechicles(self):
        all_actors = self.world.get_actors()
        return [a for a in all_actors if 'vehicle' in a.type_id]

    def _get_all_pedestrians(self):
        all_actors = self.world.get_actors()
        return [a for a in all_actors if 'pedestrian' in a.type_id]

    def try_get_new_camera_tag(self):
        """
        Try to get the new camera tag
        :return: the new camera tag
        """
        i = 0
        while True:
            tag = 'new_camera_{}'.format(i)
            if tag not in self.camera_dict.keys():
                return tag
            i += 1

    def create_rgb_camera(self,
                      width=1920, height=1080, fov=110.0,
                      x=0, y=0, z=0,
                      pitch=0, yaw=0, roll=0,
                      sensor_tick=0.04,
                      enable_postprocess_effects=True,
                      actor=None,
                      max_record_frame=1000000,
                      camera_tag="Camera0"):
        """
        Create a RGB camera to the world
        :param width: image width
        :param height: image width
        :param fov: the fov of camera
        :param x: camera location x
        :param y: camera location y
        :param z: camera location z
        :param pitch: camera rotation pitch
        :param yaw: camera rotation yaw
        :param roll: camera rotation roll
        :param sensor_tick: camera sensor tick
        :param enable_postprocess_effects: flag of postprocess effects of captured images (recommend True)
        :param actor: the actor which this camera attach to (if actor is None, then don't attach)
        :param camera_tag: camera name which also decides the save path
        :return:
        """
        if max_record_frame == 0:
            max_record_frame = sys.maxsize

        self.max_record_frame=max_record_frame

        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', str(fov))
        bp.set_attribute('sensor_tick', str(sensor_tick))
        bp.set_attribute('enable_postprocess_effects', str(enable_postprocess_effects))

        bpd = self.world.get_blueprint_library().find('sensor.camera.depth')
        bpd.set_attribute('image_size_x', str(width))
        bpd.set_attribute('image_size_y', str(height))
        bpd.set_attribute('fov', str(fov))
        bpd.set_attribute('sensor_tick', str(sensor_tick))

        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(pitch, yaw, roll)
        )

        if actor is None:
            camera = self.spawn_actor(bp, transform)
            camera_depth = self.spawn_actor(bpd, transform)
        else:
            camera = self.spawn_actor(bp, transform, attach_to=actor)
            camera_depth = self.spawn_actor(bpd, transform, attach_to=actor)

        # if not listen_method:
        #     camera.listen(self._get_listen_camera(camera_tag))

        self.actor_list.append(camera)
        self.actor_list.append(camera_depth)
        self.camera_dict[camera_tag] = SCamera(camera, camera_depth)

        return camera

    def spawn_actor(self, bp, transform, attach_to=None):
        actor = None
        while actor is None:
            time.sleep(0.1)
            if attach_to is None:
                actor = self.world.try_spawn_actor(bp, transform)
            else:
                actor = self.world.try_spawn_actor(bp, transform, attach_to=attach_to)
        return actor

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def process_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True
                elif event.key == pygame.K_TAB:
                    self.display_camera = (self.display_camera + 1) % len(self.camera_dict)
                    self.current_camera = None # remove current selected camera
                elif event.key == pygame.K_INSERT:
                    tag = self.try_get_new_camera_tag()
                    self.create_rgb_camera(camera_tag=tag)
                    self.camera_dict[tag].add_listen_queue()
                    if self.flag_show_windows:
                        self.camera_dict[tag].add_display()
                    self.display_camera = list(self.camera_dict.keys()).index(tag)
                    self.current_camera = None  # remove current selected camera
                elif event.key == pygame.K_1:
                    self.flag_image_process = not self.flag_image_process
                    self._get_current_camera().reset_depth_camera()
                elif event.key == pygame.K_2:
                    self.flag_show_3D_bbox = not self.flag_show_3D_bbox
                elif event.key == pygame.K_3:
                    self.flag_show_rect = not self.flag_show_rect
                elif event.key == pygame.K_4:
                    self.flag_show_label = not self.flag_show_label
                elif event.key == pygame.K_PLUS:
                    self.move_scale += 10
                elif event.key == pygame.K_MINUS:
                    self.move_scale -= 10
                elif event.key == pygame.K_RETURN:
                    self._save_current_cameras()
                    self._get_current_camera().save_transform(
                        os.path.join(
                            self.save_root,
                            "{}.json".format(list(self.camera_dict.keys())[self.display_camera])), self.camera_level_str[self.current_select_level])
                elif event.key == pygame.K_HOME:
                    self.current_select_level = (self.current_select_level + 1) % len(self.camera_level_str)
            elif event.type == pygame.KEYDOWN:
                if not self.flag_image_process:
                    if event.key == pygame.K_w:
                        self._get_current_camera().on_key_w_s_a_d('w', self.move_scale)
                    elif event.key == pygame.K_s:
                        self._get_current_camera().on_key_w_s_a_d('s', self.move_scale)
                    elif event.key == pygame.K_a:
                        self._get_current_camera().on_key_w_s_a_d('a', self.move_scale)
                    elif event.key == pygame.K_d:
                        self._get_current_camera().on_key_w_s_a_d('d', self.move_scale)
                    elif event.key == pygame.K_q:
                        self._get_current_camera().on_key_w_s_a_d('q', self.move_scale)
                    elif event.key == pygame.K_e:
                        self._get_current_camera().on_key_w_s_a_d('e', self.move_scale)

                    elif event.key == pygame.K_i:
                        self._get_current_camera().on_key_w_s_a_d('i', self.move_scale)
                    elif event.key == pygame.K_k:
                        self._get_current_camera().on_key_w_s_a_d('k', self.move_scale)
                    elif event.key == pygame.K_l:
                        self._get_current_camera().on_key_w_s_a_d('l', self.move_scale)
                    elif event.key == pygame.K_j:
                        self._get_current_camera().on_key_w_s_a_d('j', self.move_scale)
                    elif event.key == pygame.K_u:
                        self._get_current_camera().on_key_w_s_a_d('u', self.move_scale)
                    elif event.key == pygame.K_o:
                        self._get_current_camera().on_key_w_s_a_d('o', self.move_scale)

                    # change the weather
                    elif event.key == pygame.K_c:
                        all_weather_name = list(self.weather_conditions.keys())
                        current_index = all_weather_name.index(self.weather_name)
                        current_index = (current_index+1) / len(all_weather_name)
                        self.change_weather(all_weather_name[current_index])

        return False

    @staticmethod
    def get_font():
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)

        return pygame.font.Font(font, 14)


    def _get_all_vechicles(self):
        all_actors = self.world.get_actors()
        return [a for a in all_actors if 'vehicle' in a.type_id]

    def _get_current_camera(self):
        if self.current_camera is None:
            self.current_camera = list(self.camera_dict.values())[self.display_camera]

        return self.current_camera

    def _get_current_camera_tag(self):
        return list(self.camera_dict.keys())[self.display_camera]

    def _save_current_cameras(self):
        tag = self._get_current_camera_tag()
        # if tag not in self.config['cameras'].keys():
        self.config['cameras'].__setitem__(tag, self.camera_dict[tag].get_camera_params())
        self._save_config()

    def _save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, sort_keys=True, indent=4)
        pass

    def save_label_vehicles(self, camera_tag, image, image_depth, current_record_frame_index, flag_save_gt_data=False):
        """
        Label vehicles and save the labeled images into the save_root directories
        :param camera_tag: the spedified camera tag
        :param image: the image need to get saved
        :return: the saved image
        """
        camera = self.camera_dict[camera_tag]
        frame_number = image.frame_number
        all_vehicles = self._get_all_vechicles()
        bounding_boxes, rects, ids, physical_params, vehicle_indexes = camera.get_bounding_boxes(all_vehicles)
        overlap_ratios = camera.get_overlap_ratio(rects, image_depth)

        image = np.ndarray(
            shape=(image.height, image.width, 4),
            dtype=np.uint8,
            buffer=image.raw_data
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        for bbox, rect, id, ratio, physical_param, index in zip(bounding_boxes, rects, ids, overlap_ratios, physical_params, vehicle_indexes):

            if ratio < 0:
                continue
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                     [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            color = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
                     (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
                     (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

            camera.update_gt_data(current_record_frame_index, id, rect, bbox, ratio, physical_param, all_vehicles[index])
            # draw 3D bbox
            if self.flag_show_3D_bbox:
                for i in range(len(pairs)):
                    cv2.line(image, points[pairs[i][0]], points[pairs[i][1]], color[i], 2)

            rect = [(int(round(rect[0, 0])), int(round(rect[0, 1]))),
                    (int(round(rect[1, 0])), int(round(rect[1, 1])))]

            # draw id
            if self.flag_show_label:
                cv2.putText(image, '{}, {:.2f}'.format(id, ratio),
                            rect[0], cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255, 255, 255), 1)

            # draw rect
            if self.flag_show_rect:
                random.seed(id)
                cv2.rectangle(image,
                              rect[0],
                              rect[1],
                              (int(random.random()*255), int(random.random()*255), int(random.random()*255)), 1)

        if self.max_record_frame == sys.maxsize:
            return image
        # cv2.putText(image, str(frame_number),
        #             (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 255, 0), 2)

        # difficulty_level = ""
        # for s in self.camera_level_str:
        #     if s in camera_tag:
        #         difficulty_level = s
        #         break

        # current_camera_tag = camera_tag
        # if len(difficulty_level) > 0:
        #     current_save_root = os.path.join(current_save_root, difficulty_level)
        #     current_camera_tag = camera_tag[len(difficulty_level)+1:]

        current_save_root = os.path.join(self.save_root, self.weather_name)
        current_save_root = os.path.join(current_save_root, "{}".format(self.vehicle_num))

        save_img_path = os.path.join(os.path.join(current_save_root, 'img'), camera_tag)
        save_gt_path = os.path.join(current_save_root, "gt")
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        if not os.path.exists(save_gt_path):
            os.makedirs(save_gt_path)

        cv2.imwrite(os.path.join(save_img_path, '{}.jpg'.format(current_record_frame_index)), image)
        if flag_save_gt_data:
            camera.save_gt_data(os.path.join(save_gt_path, "{}.csv".format(camera_tag)))
        return image


    def _run(self):
        pygame.init()
        # key press hold on
        pygame.key.set_repeat(10)

        clock = pygame.time.Clock()

        frame_count = None

        font = SRecorder.get_font()

        for k in self.camera_dict.keys():
            self.camera_dict[k].add_listen_queue()
            if self.flag_show_windows:
                self.camera_dict[k].add_display()

        current_record_frame = 0
        for current_record_frame in trange(self.max_record_frame):
            # don't know why, but it works to activate the stopping vehicles.
            for v in self.world.get_actors().filter("vehicle.*"):
                v.set_autopilot(True)

            if self.process_event():
                break

            clock.tick()
            self.world.tick()

            ts = self.world.wait_for_tick()

            if frame_count is not None:
                if ts.frame_count != frame_count + 1:
                    pass

            frame_count = ts.frame_count

            if self.flag_show_windows:
                str_camera_parameters = str(self._get_current_camera().camera.get_transform())
                text_surface = font.render(str_camera_parameters, True, (0, 0, 255))
                self._get_current_camera().display.blit(text_surface, (8, 30))
                pygame.display.flip()

            if self.flag_show_windows and not self.flag_image_process:
                camera = self._get_current_camera()
                image = camera.get_image(frame_count)
                camera.display_image(image)
                show_str = "Current Camera: {}   |   ".format(self._get_current_camera_tag()) + \
                           '% 5d FPS' % clock.get_fps() + "   |   Frame: {}".format(frame_count)
                text_surface = font.render(show_str, True, (255, 255, 255))
                camera.display.blit(text_surface, (8, 10))
                pygame.display.flip()

                # show log str
                self.log_display_str = "Level: {}".format(self.camera_level_str[self.current_select_level])
                self.display_log_str(camera, font)

                continue

            if current_record_frame % 500 == 0 or self.max_record_frame - current_record_frame < 3:
                flag_save_gt_data = True
            else:
                flag_save_gt_data = False

            for i, k in enumerate(self.camera_dict.keys()):
                camera = self.camera_dict[k]
                image = camera.get_image(frame_count)
                # if i == self.display_camera:
                #     camera.display(image)

                image_depth = camera.get_image_depth(frame_count)

                image = self.save_label_vehicles(k, image, image_depth, current_record_frame, flag_save_gt_data)
                if self.flag_show_windows and i == self.display_camera:
                    camera.display_cv_image(image)
                    show_str = "Current Camera: {}   |   ".format(k) + \
                               '% 5d FPS'  % clock.get_fps() + "   |   Frame: {}".format(frame_count)
                    text_surface = font.render(show_str, True, (255, 255, 255))
                    camera.display.blit(text_surface, (8, 10))
                    pygame.display.flip()


    def display_log_str(self, camera, font):
        text_surface = font.render(self.log_display_str, True, (255, 255, 255))
        camera.display.blit(text_surface, (8, 50))
        pygame.display.flip()

    def get_recommended_colors(self):
        return ['{},{},{}'.format(random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)) for _ in range(30)]

    def can_spawn_vehicle(self, transform):
        l = transform.location
        distance = []
        for v in self.vehicle_list:
            m = v.get_transform().location
            distance += [math.sqrt((l.x-m.x)**2+(l.y-m.y)**2+(l.z-m.z)**2)]
        if len(distance) == 0 or np.min(distance) > 20:
            return True
        return False

    def try_spawn_random_vehicle_at(self, transform, blueprints):
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values*5+self.get_recommended_colors())
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')

        if self.can_spawn_vehicle(transform):
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                self.vehicle_list.append(vehicle)
                vehicle.set_autopilot()
                # print('spawned %r at %s' % (vehicle.type_id, transform.location))
                return True
        return False

    def spaw_vehicles(self, count, is_safe=True, delay=0.001):
        self.vehicle_num = count
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        if is_safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) >= 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]

        spawn_points = list(self.world.get_map().get_spawn_points())
        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            if self.try_spawn_random_vehicle_at(spawn_point, blueprints):
                count -= 1
            if count <= 0:
                break
        while count > 0:
            # time.sleep(delay)
            if self.try_spawn_random_vehicle_at(random.choice(spawn_points), blueprints):
                count -= 1

    def is_destoried(self, id):
        return self.world.get_actors().find(1) is None

    def clear_vehicles(self):
        self.clear_actors(self.vehicle_list)

    def change_weather(self, name):
        """
        Change the weather by the name (support "Rain, "Cloudy", "Clear")
        :param name: A weather selected from "Rain, "Cloudy" or "Clear".
        :return: None
        """
        self.world.set_weather(self.random_get_weather(name))

    def clear_actors(self, actor_list):
        self.client.apply_batch([carla.command.DestroyActor(x.id) for x in actor_list if x is not None])
        time.sleep(2)

        count = np.sum([self.is_destoried(a.id) for a in actor_list])
        while count > 0:
            self.client.apply_batch([carla.command.DestroyActor(x.id) for x in actor_list if not self.is_destoried(x)])
            time.sleep(2)
            count = np.sum([self.is_destoried(a.id) for a in actor_list])
        actor_list.clear()



    def clear(self):
        print("clear vehicles and cameras >>>>>>>>>>>>>>>>>>>>>..")
        for c in self.camera_dict.values():
            c.stop()

        self.clear_actors(self.actor_list)
        self.clear_actors(self.vehicle_list)
        self.camera_dict.clear()

    def run(self):
        try:
            self.set_synchronous_mode(True)
            self._run()
        finally:
            self.set_synchronous_mode(False)
            self.clear()

if __name__ == "__main__":
    recorder = SRecorder()
    recorder.create_rgb_camera(
        width=1920,
        height=1080,
        z=25,
        fov=90,
        pitch=-40,
        yaw=4,
        roll=5,
        camera_tag="Camera0"
    )

    recorder.create_rgb_camera(
        width=1920,
        height=1080,
        z=50,
        fov=90,
        pitch=-40,
        yaw=4,
        roll=5,
        camera_tag="Camera1"
    )

    recorder.run()

