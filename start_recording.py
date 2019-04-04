#  Copyright (c) 2019. ShiJie Sun at the Chang'an University
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#  Author: shijie Sun
#  Email: shijieSun@chd.edu.cn
#  Github: www.github.com/shijieS

from recording_utils import SRecorder
import argparse
from config import Configure

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

argparser = argparse.ArgumentParser(
        description='Awesome recording script')

argparser.add_argument(
        '--config_name',
        default='test.json',
        help='configure file name')
argparser.add_argument(
    '--recording_num_scale',
    default=0.5,
    type=float,
    help='the scale of recording frame, default is 0.5, which means 0.5*10000'
)

argparser.add_argument(
    '--auto_save',
    default=True,
    type=str2bool,
    help='whether show display windows or not'
)

argparser.add_argument(
    '--flag_show_windows',
    default=False,
    type=str2bool,
    help='whether show display windows or not'
)

argparser.add_argument(
    '--port',
    default=2000,
    type=int,
    help='server port'
)

from tqdm import trange

args = argparser.parse_args()

def start_recording(config_name, recording_num_scale, flag_show_windows, auto_save, port=None):
    config = Configure.get_config(config_name)
    if port is None:
        port = config['port']

    cameras = config['cameras']
    vehicle_nums = config['vehicle_num']
    weathers = config['weathers']

    all_conditions = [(w, v) for w in weathers for v in vehicle_nums]

    if config["mode"] == "parallel":
        for cond in all_conditions:
            recorder = SRecorder(host=config['host'],
                                 port=port,
                                 save_root=config['save_root'],
                                 weather_name=cond[0],
                                 vehicle_num=cond[1],
                                 flag_show_windows=flag_show_windows,
                                 auto_save=auto_save
                                 )
            camera_keys = cameras.keys()

            for _, k in zip(trange(len(cameras)), cameras.keys()):
                recorder.create_rgb_camera(
                    width=cameras[k]["width"],
                    height=cameras[k]["height"],
                    x=cameras[k]["x"],
                    y=cameras[k]["y"],
                    z=cameras[k]["z"],
                    fov=cameras[k]["fov"],
                    pitch=cameras[k]["pitch"],
                    yaw=cameras[k]["yaw"],
                    roll=cameras[k]["roll"],
                    max_record_frame=int(cameras[k]["max_record_frame"]*recording_num_scale),
                    camera_tag=k
                )
            recorder.run()

    elif config["mode"] == "serial":
        for cond in all_conditions:
            for k in cameras.keys():
                recorder = SRecorder(host=config['host'],
                                     port=port,
                                     save_root=config['save_root'],
                                     weather_name=cond[0],
                                     vehicle_num=cond[1],
                                     flag_show_windows=flag_show_windows,
                                     auto_save=auto_save
                                     )
                recorder.create_rgb_camera(
                    width=cameras[k]["width"],
                    height=cameras[k]["height"],
                    x=cameras[k]["x"],
                    y=cameras[k]["y"],
                    z=cameras[k]["z"],
                    fov=cameras[k]["fov"],
                    pitch=cameras[k]["pitch"],
                    yaw=cameras[k]["yaw"],
                    roll=cameras[k]["roll"],
                    max_record_frame=int(cameras[k]["max_record_frame"]*recording_num_scale),
                    camera_tag=k
                )
                recorder.run()


if __name__ == "__main__":
    start_recording(args.config_name, args.recording_num_scale,
                    args.flag_show_windows, args.auto_save, args.port)