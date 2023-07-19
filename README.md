# Neural Haircut: Prior-Guided Strand-Based Hair Reconstruction



[**Paper**](https://arxiv.org/abs/2306.05872) | [**Project Page**](https://samsunglabs.github.io/NeuralHaircut/)



This repository contains official inference code for Neural Haircut.

This code helps you to create strand-based hairstyle using multi-view images or monocular video. 




## Getting started


Clone the repository and install requirements: 

```bash
git clone https://github.sec.samsung.net/v-sklyarova/NeuralHaircut.git
cd NeuralHaircut
conda env create -n neuralhaircut -f neural_haircut.yaml
conda activate neuralhaircut
```

Initialize submodules of [k-diffusion](https://github.com/crowsonkb/k-diffusion), [NeuS](https://github.com/Totoro97/NeuS), [MODNet](https://github.com/ZHKKKe/MODNet), [CDGNet](https://github.com/tjpulkl/CDGNet), [npbgpp](https://github.com/rakhimovv/npbgpp). Download pretrained weights for [CDGNet](https://github.com/tjpulkl/CDGNet) and [MODNet](https://github.com/ZHKKKe/MODNet).

```bash
git submodule update --init --recursive
```


```bash
cd npbgpp && python setup.py build develop
cd ..
```


Download the pretrained NeuralHaircut models:

```bash
gdown --folder https://drive.google.com/drive/folders/1TCdJ0CKR3Q6LviovndOkJaKm8S1T9F_8
```

## Running

### Fitting the FLAME coarse geometry using multiview images

More details could be find in [multiview_optimization](./src/multiview_optimization)


### Launching the first stage on [H3ds dataset](https://github.com/CrisalixSA/h3ds) or custom monocular dataset:



```bash
python run_geometry_reconstruction.py --case CASE --conf ./configs/SCENE_TYPE/neural_strands.yaml --exp_name first_stage_SCENE_TYPE_CASE
```

where ```SCENE_TYPE = [h3ds|monocular]```.



- If you want to add camera fitting: 

```bash
python run_geometry_reconstruction.py --case CASE --conf ./configs/SCENE_TYPE/neural_strands_w_camera_fitting.yaml --exp_name first_stage_SCENE_TYPE_CASE --train_cameras
```

At the end of first stage do the [following steps](./preprocess_custom_data).


- If you want to continue from checkpoint add flag ```--is_continue```.

- If you want to obtain mesh in higher resolution add flags ```--is_continue --mode validate_mesh```.



### Launching the second stage on [H3ds dataset](https://github.com/CrisalixSA/h3ds) or custom monocular dataset:


```bash
python run_strands_optimization.py --case CASE --scene_type SCENE_TYPE --conf ./configs/SCENE_TYPE/neural_strands.yaml  --hair_conf ./configs/hair_strands_textured.yaml --exp_name second_stage_SCENE_TYPE_CASE
```

- If during the first stage you also fitted the cameras, then use the following:

```bash
python run_strands_optimization.py --case CASE --scene_type SCENE_TYPE --conf ./configs/SCENE_TYPE/neural_strands_w_camera_fitted.yaml  --hair_conf ./configs/hair_strands_textured.yaml --exp_name second_stage_SCENE_TYPE_CASE
```


## Train NeuralHaircut with your custom data

More information can be found in [preprocess_custom_data.](./preprocess_custom_data).

You could run the scripts on our [monocular scene](./example) for convenience. 


### License

This code and model are available for scientific research purposes as defined in the LICENSE.txt file. 
By downloading and using the project you agree to the terms in the LICENSE.txt.


## Links

This work is based on the great project [NeuS](https://github.com/Totoro97/NeuS). Also we acknowledge additional projects that were essential and speed up the developement.

- [NeuS](https://github.com/Totoro97/NeuS) for geometry reconstruction;

- [npbgpp](https://github.com/rakhimovv/npbgpp) for rendering of soft rasterized features;

- [k-diffusion](https://github.com/crowsonkb/k-diffusion) for diffusion network;

- [MODNet](https://github.com/ZHKKKe/MODNet), [CDGNet](https://github.com/tjpulkl/CDGNet) used to obtain silhouette and hair segmentations;

- [PIXIE](https://github.com/yfeng95/PIXIE) used to obtain initialization for shape and pose parameters;

## Citation


Cite as below if you find this repository is helpful to your project:

```
@inproceedings{sklyarova2023neural_haircut,
title = {Neural Haircut: Prior-Guided Strand-Based Hair Reconstruction},
author = {Sklyarova, Vanessa and Chelishev, Jenya and Dogaru, Andreea and Medvedev, Igor and Lempitsky, Victor and Zakharov, Egor},
booktitle = {Proceedings of IEEE International Conference on Computer Vision (ICCV)},
year = {2023}
} 
```






