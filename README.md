# Awesome Multiple Object Tracking Dataset Recorder
> Here, we publish an awesome multiple object tracker recorder

## Requirement
- CARLA 0.9.4
- \>= Ubuntu 16.0

## Usage
### Short Cut Key
| Key    | Action                              |
|--------|-------------------------------------|
| Esc    | Quit                                |
| Tab    | Switch the Cameras                   |
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