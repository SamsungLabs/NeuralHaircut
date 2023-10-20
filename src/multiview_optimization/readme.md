Download pre-trained models and data from [SMPLX](https://smpl-x.is.tue.mpg.de/)  and [PIXIE](https://pixie.is.tue.mpg.de/) projects.

For more information please follow [PIXIE installation](https://github.com/yfeng95/PIXIE/blob/master/Doc/docs/getting_started.md).

For multiview optimization you need to have the following files:
- `SMPL-X__FLAME_vertex_ids.npy`: Download *"MANO and FLAME vertex indices"* from [SMPLX download page](https://smpl-x.is.tue.mpg.de/download.php)
- `smplx_extra_joints.yaml`: Download *"SHAPY Model"* from [SHAPY download page](https://shapy.is.tue.mpg.de/download.php)
- `SMPLX_NEUTRAL_2020.npz`: Download *"SMPL-X 2020"* from [SMPLX download page](https://smpl-x.is.tue.mpg.de/download.php)

and put these files into `./PIXIE/data`.

Note, that you need to obtain  [PIXIE initialization](https://github.com/yfeng95/PIXIE) for shape, pose parameters and save it as a dict in ```initialization_pixie```.
Specifically, assuming that the case name is `person_1` and the image data is stored in `NeuralHaircut/implicit-hair-data/data/monocular/person_1`, run PIXIE on the images:

```bash
cd <path/to/PIXIE>
python demos/demo_fit_face.py -i <path/to/NeuralHaircut>/implicit-hair-data/data/monocular/person_1/image -s <path/to/NeuralHaircut>/implicit-hair-data/data/monocular/person_1/pixie --saveParam True
```
Then, run:
```
python ./concat_pixie.py --case person_1 --scene_type monocular
```
which will produce `NeuralHaircut/implicit-hair-data/data/monocular/person_1/initialization_pixie`.


Furthermore, use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to obtain 3d keypoints:
```bash
cd <path/to/openpose>
./build/examples/openpose/openpose.bin -display 0 -render_pose 0 -face -hand -image_dir <path/to/NeuralHaircut>/implicit-hair-data/data/monocular/person_1/image  -write_json <path/to/NeuralHaircut>/implicit-hair-data/data/monocular/person_1/openpose_kp
```


To obtain FLAME prior run:

```bash
bash run_monocular_fitting.sh
```
To visualize the training process:

```bash
tensorboard --logdir ./experiments/EXP_NAME
```

After training put obtained FLAME prior mesh (i.e., the last `*.obj` in `./experiments/fit_person_1_bs_20_train_rot_shape/mesh`) into the dataset folder ```./implicit-hair-data/data/monocular/person_1/head_prior.obj```.
