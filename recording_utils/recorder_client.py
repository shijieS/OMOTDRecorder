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


class SRecorder:
    def __init__(self, host='localhost', port=2000,
                 save_root='./recorders/'):
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
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.actor_list = []
        self.camera_dict = OrderedDict()
        self.display_camera = 0
        self.current_camera = None

        self.flag_show_3D_bbox = False
        self.flag_show_rect = False
        self.flag_show_label = False
        self.flag_image_process = False

        self.move_scale = 30

    def _get_listen_camera(self, camera_tag="camera0"):
        def listen(image):
            frame_number = image.frame_number

            self.world.wait_for_tick()

            all_vehicles = self._get_all_vechicles()
            camera = self.camera_dict[camera_tag]
            bounding_boxes = camera.get_bounding_boxes(all_vehicles)

            image = np.ndarray(
                shape=(image.height, image.width, 4),
                dtype=np.uint8,
                buffer=image.raw_data
            )

            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            for bbox in bounding_boxes:
                points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

                pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                         [4, 5], [5, 6], [6, 7], [7, 4],
                         [0, 4], [1, 5], [2, 6], [3, 7]]
                color = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
                         (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
                         (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

                for i in range(len(pairs)):
                    cv2.line(image, points[pairs[i][0]], points[pairs[i][1]], color[i])

            cv2.putText(image, str(frame_number),
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            save_path = os.path.join(self.save_root, camera_tag)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(frame_number)), image)

        return listen

    def _get_all_vechicles(self):
        all_actors = self.world.get_actors()
        return [a for a in all_actors if 'vehicle' in a.type_id]

    def _get_all_pedestrians(self):
        all_actors = self.world.get_actors()
        return [a for a in all_actors if 'pedestrian' in a.type_id]

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
            camera = self.world.try_spawn_actor(bp, transform)
            camera_depth = self.world.try_spawn_actor(bpd, transform)
        else:
            camera = self.world.try_spawn_actor(bp, transform, attach_to=actor)
            camera_depth = self.world.try_spawn_actor(bpd, transform, attach_to=actor)

        # if not listen_method:
        #     camera.listen(self._get_listen_camera(camera_tag))

        self.actor_list.append(camera)
        self.actor_list.append(camera_depth)
        self.camera_dict[camera_tag] = SCamera(camera, camera_depth)

        return camera

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
                elif event.key == pygame.K_INSERT:
                    self._get_current_camera().save_transform(
                        os.path.join(
                            self.save_root,
                            "{}.json".format(list(self.camera_dict.keys())[self.display_camera])))
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

    def save_label_vehicles(self, camera_tag, image, image_depth):
        """
        Label vehicles and save the labeled images into the save_root directories
        :param camera_tag: the spedified camera tag
        :param image: the image need to get saved
        :return: the saved image
        """
        camera = self.camera_dict[camera_tag]
        frame_number = image.frame_number
        all_vehicles = self._get_all_vechicles()
        bounding_boxes, rects, ids = camera.get_bounding_boxes(all_vehicles)
        overlap_ratios = camera.get_overlap_ratio(rects, image_depth)

        image = np.ndarray(
            shape=(image.height, image.width, 4),
            dtype=np.uint8,
            buffer=image.raw_data
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        for bbox, rect, id, ratio in zip(bounding_boxes, rects, ids, overlap_ratios):
            if ratio < 0:
                continue
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                     [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            color = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
                     (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
                     (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

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

        # cv2.putText(image, str(frame_number),
        #             (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 255, 0), 2)


        save_path = os.path.join(self.save_root, camera_tag)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(frame_number)), image)
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
            self.camera_dict[k].add_display()

        current_record_frame = 0
        while current_record_frame < self.max_record_frame:
            current_record_frame += 1
            if self.process_event():
                break

            clock.tick()
            self.world.tick()

            ts = self.world.wait_for_tick()

            if frame_count is not None:
                if ts.frame_count != frame_count + 1:
                    print('frame skip!')

            frame_count = ts.frame_count

            str_camera_parameters = str(self._get_current_camera().camera.get_transform())
            text_surface = font.render(str_camera_parameters, True, (0, 0, 255))
            self._get_current_camera().display.blit(text_surface, (8, 30))
            pygame.display.flip()

            if not self.flag_image_process:
                camera = self._get_current_camera()
                image = camera.get_image(frame_count)
                camera.display_image(image)
                show_str = "Current Camera: {}   |   ".format(k) + \
                           '% 5d FPS' % clock.get_fps() + "   |   Frame: {}".format(frame_count)
                text_surface = font.render(show_str, True, (255, 255, 255))
                camera.display.blit(text_surface, (8, 10))
                pygame.display.flip()

                continue

            for i, k in enumerate(self.camera_dict.keys()):
                camera = self.camera_dict[k]
                image = camera.get_image(frame_count)
                # if i == self.display_camera:
                #     camera.display(image)

                image_depth = camera.get_image_depth(frame_count)

                image = self.save_label_vehicles(k, image, image_depth)
                if i == self.display_camera:
                    camera.display_cv_image(image)
                    show_str = "Current Camera: {}   |   ".format(k) + \
                               '% 5d FPS'  % clock.get_fps() + "   |   Frame: {}".format(frame_count)
                    text_surface = font.render(show_str, True, (255, 255, 255))
                    camera.display.blit(text_surface, (8, 10))
                    pygame.display.flip()


    def clear(self):
        self.client.apply_batch([carla.command.DestroyActor(x.id) for x in self.actor_list])

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

