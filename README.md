# Awesome Multiple Object Tracking Dataset Recorder
> Here, we publish an awesome multiple object tracker recorder

## Requirement
- CARLA 0.9.4
- \>= Ubuntu 16.0

## Usage

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