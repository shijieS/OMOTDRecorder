echo "==================Start Create Simulator=================="
x-terminal-emulator -e sh -c "export CUDA_VISIBLE_DEVICES=0; cd /home/ssj/Data/github/CARLA_0.9.4; ./CarlaUE4.sh Town02 -carla-server-timeout=500000ms -quality-level=Epic -carla-port=2000 -benchmark -fps=30"
x-terminal-emulator -e sh -c "export CUDA_VISIBLE_DEVICES=1; cd /home/ssj/Data/github/CARLA_0.9.4; ./CarlaUE4.sh Town04 -carla-server-timeout=500000ms -quality-level=Epic -carla-port=2010 -benchmark -fps=30"
x-terminal-emulator -e sh -c "export CUDA_VISIBLE_DEVICES=2; cd /home/ssj/Data/github/CARLA_0.9.4; ./CarlaUE4.sh Town05 -carla-server-timeout=500000ms -quality-level=Epic -carla-port=2020 -benchmark -fps=30"

echo "==================Successfully Create Simulator=================="
echo "==================Wait ....=================="
sleep 20

echo "==================Start Recording Script=================="


x-terminal-emulator -e sh -c "cd /home/ssj/Data/github/AMOTDRecorder; source activate CARLA; /home/ssj/miniconda3/envs/CARLA/bin/python start_recording.py --port=2000 --config_name=Town02_all.json --recording_num_scale=0.5"
x-terminal-emulator -e sh -c "cd /home/ssj/Data/github/AMOTDRecorder; source activate CARLA; /home/ssj/miniconda3/envs/CARLA/bin/python start_recording.py --port=2010 --config_name=Town04_all.json --recording_num_scale=0.5"
x-terminal-emulator -e sh -c "cd /home/ssj/Data/github/AMOTDRecorder; source activate CARLA; /home/ssj/miniconda3/envs/CARLA/bin/python start_recording.py --port=2020 --config_name=Town05_all.json --recording_num_scale=0.5"

echo "==================Well Done!!=================="