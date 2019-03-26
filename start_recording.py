from recording_utils import SRecorder



if __name__ == "__main__":
    recorder = SRecorder(save_root='./data_recorders/')
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
        z=3,
        fov=90,
        pitch=-0,
        yaw=4,
        roll=5,
        camera_tag="Camera1"
    )

    recorder.run()