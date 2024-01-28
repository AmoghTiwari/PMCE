#!/bin/bash
python ./main/run_demo.py --vid_file /data/amogh/projects/smpl_fitting_tcs/data/20240117_For_Comparison/new/Human3_6M/S11/_ALL_1.60457274.mp4 --gpu 1 --save_pkl
python ./main/run_demo.py --vid_file /data/amogh/projects/smpl_fitting_tcs/data/20240117_For_Comparison/new/3DPW/courtyard_drinking_00.mp4 --gpu 1 --save_pkl
python ./main/run_demo.py --vid_file /data/amogh/projects/smpl_fitting_tcs/data/20240117_For_Comparison/new/3DPW/downtown_downstairs_00.mp4 --gpu 1 --save_pkl
python ./main/run_demo.py --vid_file /data/amogh/projects/smpl_fitting_tcs/data/20240117_For_Comparison/new/3DPW/outdoors_fencing_01.mp4 --gpu 1 --save_pkl

