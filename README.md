# Omni-MOT Dataset Recorder
> This is the repository of recording [Omni-MOT]() dataset. Its functions include: 
>
> - Camera Operations: Move forward backward, pitch up down, roll left right, yaw left right
> - Save camera parameters
> - Calculate the ground truth data for multiple object tracking (i.e. 3D bounding boxes, 2D bounding boxes, Visibility)
> - Save Videos and Ground Truth

## Requirement
- CARLA 0.9.4

- \>= Ubuntu 16.0

- Install the required packages

  ```shell
  cd <project>
  pip install -r requirement.txt
  ```

## Usage

### Open
```shell
cd <CARLA Project>
source activate CARLA
./CarlaUE4.sh
```

### Short Cut Key
| Key    | Action                              |
|--------|-------------------------------------|
| Esc    | Quit                                |
| Tab    | Switch the Cameras                  |
| 1      | Start Image Processing              |
| 2      | Show 3D Bounding Boxes in the Image |
| 3      | Show Rectangle in the Image         |
| 4      | Show Vehicle Labels                 |
| w      | Move Forward                        |
| s      | Move Backward                       |
| a      | Move Left                           |
| d      | Move Right                          |
| q      | Move Up                             |
| e      | Move Down                           |
| i      | Pitch Up                            |
| k      | Pitch Down                          |
| j      | Roll Left                           |
| l      | Roll Right                          |
| u      | Yaw Left                            |
| p      | Yaw Right                           |
| Insert | Save Camera Parameter               |
| Home   | Update the Camera Level             |

### Configure Script
The following is the configure script:

```json
{
  "mode": "parallel",
  "save_root": "/home/ssj/Data/github/AwesomeMOTDataset/Dataset",
  "host": "127.0.0.1",
  "port": 2000,
  "cameras": {
    "Camera0":{
      "width": 1920,
      "height": 1080,
      "x": 0,
      "y": 0,
      "z": 25,
      "fov": 90,
      "pitch": -40,
      "yaw": 4,
      "roll": 5,
      "max_record_frame": 10000
    },

    "Camera1": {
      "width": 1920,
      "height": 1080,
      "x": 0,
      "y": 0,
      "z": 3,
      "fov": 90,
      "pitch": -0,
      "yaw": 4,
      "roll": 5,
      "max_record_frame": 10000
    }
  }
}
```

### Video Format
**Xvid**