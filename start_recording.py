from recording_utils import SRecorder
import argparse
from config import Configure

argparser = argparse.ArgumentParser(
        description='Awesome recording script')

argparser.add_argument(
        '--config_name',
        default='town1.json',
        help='configure file name')

args = argparser.parse_args()

def start_recording(config_name):
    config = Configure.get_config(config_name)

    cameras = config['cameras']

    if config["mode"] == "parallel":
        recorder = SRecorder(host=config['host'],
                             port=config['port'],
                             save_root=config['save_root']
                             )
        for k in cameras.keys():
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
                max_record_frame=cameras[k]["max_record_frame"],
                camera_tag=k
            )
        recorder.run()

    elif config["mode"] == "serial":
        for k in cameras.keys():
            recorder = SRecorder(host=config['host'],
                                 port=config['port'],
                                 save_root=config['save_root']
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
                max_record_frame=cameras[k]["max_record_frame"],
                camera_tag=k
            )
            recorder.run()


if __name__ == "__main__":
    start_recording(args.config_name)