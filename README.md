# gSDF: Geometry-Driven Signed Distance Functions for 3D Hand-Object Reconstruction (CVPR 2023)

This repository is the official implementation of [gSDF: Geometry-Driven Signed Distance Functions for 3D Hand-Object Reconstruction](https://arxiv.org/abs/2304.11970). It also provides implementations for [Grasping Field](https://arxiv.org/abs/2008.04451) and [AlignSDF](https://arxiv.org/abs/2207.12909). 
Project webpage: https://zerchen.github.io/projects/gsdf.html

Abstract: Signed distance functions (SDFs) is an attractive framework that has recently shown promising results for 3D shape reconstruction from images. SDFs seamlessly generalize to different shape resolutions and topologies but lack explicit modelling of the underlying 3D geometry. In this work, we exploit the hand structure and use it as guidance for SDF-based shape reconstruction. In particular, we address reconstruction of hands and manipulated objects from monocular RGB images. To this end, we estimate poses of hands and objects and use them to guide 3D reconstruction. More specifically, we predict kinematic chains of pose transformations and align SDFs with highly-articulated hand poses. We improve the visual features of 3D points with geometry alignment and further leverage temporal information to enhance the robustness to occlusion and motion blurs. We conduct extensive experiments on the challenging ObMan and DexYCB benchmarks and demonstrate significant improvements of the proposed method over the state of the art.

## Installation
Please follow instructions listed below to build the environment.
```
conda create -n gsdf python=3.9
conda activate gsdf
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
## Dataset
1. ObMan dataset preparations. 
- Download ObMan data from [the official website](https://www.di.ens.fr/willow/research/obman/data/requestaccess.php).
- Set up a soft link from the download path to `${ROOT}/datasets/obman/data`.
- Download processed [SDF files](https://drive.google.com/drive/folders/1GjFJBJlbJxeYrExtcYEdhAaeH-wLZOIF) and [json files](https://drive.google.com/drive/folders/1DBzG9J0uLzCy4A6W6Uq6Aq4JNAHiiNJQ).
- Run `${ROOT}/preprocess/cocoify_obman.py` to generate LMDB training files. The data organization looks like this: 
   ```
   ${ROOT}/datasets/obman
   └── splits
       obman_train.json
       obman_test.json
       obman.py
       data
        ├── val
        ├── train
        |   ├── rgb
        |   ├── rgb.lmdb
        |   ├── sdf_hand
        |   ├── sdf_hand.lmdb
        |   ├── sdf_obj
        |   ├── sdf_obj.lmdb
        └── test
            ├── rgb
            ├── mesh_hand
            ├── mesh_obj
   ```

2. DexYCB dataset preparations. 
- Download DexYCB data from [the official webpage](https://dex-ycb.github.io/).
- Set up a soft link from the download path to `${ROOT}/datasets/dexycb/data`.
- Download processed [SDF files](https://drive.google.com/drive/folders/15yjzjYcqyOiIbX-6uaeYOezVH4stDTCG) and [json files](https://drive.google.com/drive/folders/1qULhMx1PrnXkihrPacIFzLOT5H2FZSj7).
- Run `${ROOT}/preprocess/cocoify_dexycb.py` to generate LMDB training files. The data organization looks like this: 
   ```
   ${ROOT}/datasets/obman
   └── splits
       toolkit
       dexycb_train_s0.json
       dexycb_test_s0.json
       dexycb.py
       data
        ├── 20200709-subject-01
        ├── .
        ├── .
        ├── 20201022-subject-10
        ├── bop
        ├── models
        ├── mesh_data
        ├── sdf_data
        ├── rgb_s0.lmdb
        ├── sdf_hand_s0.lmdb
        └── sdf_obj_s0.lmdb
   ```

## Training
1. Establish the output directory by `mkdir ${ROOT}/outputs` and `cd ${ROOT}/tools`.
2. `${ROOT}/playground` provides implementations of different models:
   ```
   ${ROOT}/playground
    ├── pose_kpt                  # A component for gSDF which solves pose estimation problem
    ├── hsdf_osdf_1net            # The SDF network with a single backbone like Grasping Field or AlignSDF
    ├── hsdf_osdf_2net            # The SDF network with two backbones like gSDF
    ├── hsdf_osdf_2net_pa         # Compared with hsdf_osdf_2net, it additionally uses pixel-aligned visual features
    ├── hsdf_osdf_2net_video_pa   # Compared with hsdf_osdf_2net_pa, it additionally uses spatial-temporay transformer to process multiple frames
   ```

2. Train the Grasping Field model:
```
bash dist_train.sh 4 1234 -e ../playground/hsdf_osdf_1net/experiments/obman_resnet18_hnerf3_onerf3.yaml --gpu 0-3 use_lmdb True
```
4. Train the AlignSDF model:
```
bash dist_train.sh 4 1234 -e ../playground/hsdf_osdf_1net/experiments/obman_resnet18_hkine6_otrans6.yaml --gpu 0-3 use_lmdb True
```

5. Train the gSDF model:
```
# It first needs to train a checkpoint for hand pose estimation.
bash dist_train.sh 4 1234 -e ../playground/pose_kpt/experiments/obman_hand.yaml --gpu 0-3 use_lmdb True

# Then, load the pretrained pose checkpoint and train the SDF model.
bash dist_train.sh 4 1234 -e ../playground/hsdf_osdf_2net_pa/experiments/obman_presnet18_sresnet18_hkine6_okine6.yaml --gpu 0-3 use_lmdb True hand_point_latent 51 obj_point_latent 72 ckpt path_to_pretrained_model

# Train the model that processes multiple frames (DexYCB provides videos).
bash dist_train.sh 4 1234 -e ../playground/hsdf_osdf_2net_video_pa/experiments/dexycbs0_3frames_presnet18_sresnet18_hkine6_okine6.yaml --gpu 0-3 use_lmdb True hand_point_latent 51 obj_point_latent 72 ckpt path_to_pretrained_model
```

## Testing and Evaluation
Actually, when it finishes training, the script will launch the testing automatically. You could also launch the training explicitly by:
```
bash dist_test.sh 4 1234 -e ../outputs/exp_name/somename.cfg --gpu 0-3
```
After the testing phase ends, you could evaluate the performance:
```
python eval.py -e ../outputs/exp_name/
```

## Citation
If you find this work useful, please consider citing:
```
@InProceedings{chen2023gsdf,
author       = {Chen, Zerui and Chen, Shizhe and Schmid, Cordelia and Laptev, Ivan},
title        = {{gSDF}: {Geometry-Driven} Signed Distance Functions for {3D} Hand-Object Reconstruction},
booktitle    = {CVPR},
year         = {2023},
}
```

## Acknowledgement
Some of the codes are built upon [manopth](https://github.com/hassony2/manopth), [PoseNet](https://github.com/mks0601/3DMPPE_POSENET_RELEASE), [PCL](https://github.com/yu-frank/PerspectiveCropLayers) [Grasping Field](https://github.com/korrawe/grasping_field), and [HALO](https://github.com/korrawe/halo).
Thanks them for their great works!