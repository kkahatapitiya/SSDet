# SSDet
Code for our AAAI 2023 paper "Weakly-guided Self-supervised Pretraining for Temporal Activity Detection"

Code to focus are marked within:
----------------
########## START ###########
########### END ############


Volume Augmentations implemented in:
------------------------------------
./x3d.py


Soft-label based loss implemented in:
-------------------------------------
./train_x3d_kinetics_multigrid.py


To run the code:
----------------
setup the data directories for kinetics and charades, then run,
python train_x3d_kinetics_multigrid.py -gpu 0,1     # train on kinetics using volume augmentations
python train_x3d_charades_loc.py -gpu 0,1           # finetune on downstream charades using above checkpoint


This project is based on the work at:
https://github.com/facebookresearch/SlowFast/tree/main/projects/x3d
https://github.com/facebookresearch/SlowFast/tree/main/projects/multigrid
