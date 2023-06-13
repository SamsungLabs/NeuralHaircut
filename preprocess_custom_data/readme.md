### Data structure

The full data folder is organized as follows:


```
|-- NeuralHaircut/implicit-hair-data
    |-- data
        |-- h3ds
        |-- monocular
            |-- case_name
                |-- video_frames  # after parsing .mp4 (optional)
                |-- colmap # (optional) 
                    |-- full_res_image
                    |-- cameras.npz
                    |-- point_cloud.ply
                    |-- database.db
                    |-- sparse
                    |-- dense
                    |-- sparse_txt

                |-- cameras.npz    # camera parameters
                |-- image
                |-- mask
                |-- hair_mask
                |-- orientation_maps
                |-- confidence_maps
                |-- dif_mask.png # scalp masking for diffusion model
                |-- cut_scalp_verts.pickle # scalp vertex for hairstyle
                |-- head_prior.obj  # FLAME prior head
                |-- head_prior_wo_eyes.obj # version wo eyes
                |-- scale.pickle # scale the scene into unit sphere
                |-- views.pickle # index of chosen views (optional)
                |-- initialization_pixie # initialization for shape, expression, pose, ...
                |-- openpose_kp # needed for mesh prior fitting (optional)   
                |-- fitted_cameras.pth # Checkpoint for fitted cameras (optional)

```


### For the first stage you need the following:



#### Step 1. (Optional) Run [COLMAP SfM](https://colmap.github.io/) to obtain cameras. 

##### Run commands

```bash
colmap automatic_reconstructor --workspace_path  CASE_NAME/colmap  --image_path CASE_NAME/video_frames
```

```bash
mkdir CASE_NAME/colmap/sparse_txt && colmap model_converter --input_path CASE_NAME/colmap/sparse/0  --output_path CASE_NAME/colmap/sparse_txt --output_type TXT
```



##### To postprocess COLMAP's output run:

```bash
python colmap_parsing.py --path_to_scene  ./implicit-hair-data/data/SCENE_TYPE/CASE --save_path ./implicit-hair-data/data/SCENE_TYPE/CASE/colmap
```
##### Obtain:

After this step you would obtain ```colmap/full_res_image, colmap/cameras.npz, colmap/point_cloud.ply```


#### Step 2.  (Optional) Define the region of interests in obtained point cloud.

Obtained ```colmap/point_cloud.ply``` is very noisy, so we are additionally define the region of interest using MeshLab and upload it to the current folder as ```point_cloud_cropped.ply```.


#### Step 3. Transform cropped scene to lie in a unit sphere volume.

```bash
python preprocess_custom_data/scale_scene_into_sphere.py --case CASE --scene_type SCENE_TYPE --path_to_data ./implicit-hair-data/data/
```
After this step in```./implicit-hair-data/data/SCENE_TYPE/CASE``` you would obtain ```scale.pickle```.


#### Step 4. (Optional) Crop input images and postprocess cameras. 

Note, now NeuralHaircut supports only the square images.

#### Step 5. Obtain hair, silhouette masks and orientation and confidence maps.


```bash
python preprocess_custom_data/calc_masks.py --scene_path ./implicit-hair-data/data/SCENE_TYPE/CASE/ --MODNET_ckpt path_to_modnet --CDGNET_ckpt path_to_cdgnet
```


```bash
python preprocess_custom_data/calc_orientation_maps.py --img_path ./implicit-hair-data/data/SCENE_TYPE/CASE/image/ --orient_dir ./implicit-hair-data/data/SCENE_TYPE/CASE/orientation_maps --conf_dir ./implicit-hair-data/data/SCENE_TYPE/CASE/confidence_maps
```

After this step in```./implicit-hair-data/data/SCENE_TYPE/CASE``` you would obtain ```hair_mask, mask, confidence_maps, orientation_maps```.


#### Step 6. (Optional) Define views on which you want to train  and save it into views.pickle file.



#### Step 7. Using multiview images and cameras obtain FLAME head.

FLAME head would be used to regularize the scalp region.

More details could be find in [multiview_optimization](../src/multiview_optimization)

#### Step 8. Cut eyes of FLAME head, needed for scalp regularizaton.

Could use MeshLab or run the following with predefined eyes faces.

```bash
python  ./preprocess_custom_data/cut_eyes.py --case CASE --scene_type SCENE_TYPE --path_to_data ./implicit-hair-data/data/
```

After this step in```./implicit-hair-data/data/SCENE_TYPE/CASE``` you would obtain ```head_prior_wo_eyes.obj```.


### For the second stage you need the following:

#### Step 1. Copy the checkpoint for hair sdf and orientation field, obtained meshes to the scene folder for convenience; 

```bash
python ./preprocess_custom_data/copy_checkpoints.py --case CASE --exp_name first_stage_reconctruction_CASE --conf_path ./configs/SCENE_TYPE/neural_strands*.yaml
```
```* use neural_strands_w_camera_fitting.yaml``` if train with camera fitting.

After this step in```./implicit-hair-data/data/SCENE_TYPE/CASE``` you would obtain ```final_hair.ply, final_head.ply, ckpt_final.pth, fitted_cameras.pth (optional)```

#### Step 2. Extract visible hair surface from sdf;

```bash
python ./preprocess_custom_data/extract_visible_surface.py --conf_path ./configs/SCENE_TYPE/neural_strands*.yaml  --case CASE --scene_type SCENE_TYPE --img_size 2160 --n_views 2
```

After this step in ```./implicit-hair-data/data/SCENE_TYPE/CASE``` you would obtain ```hair_outer.ply```.

#### Step 3. Remesh hair_outer.ply to ~10k vertex for acceleration;

Note, you could use either Meshlab to do that or any other library. Also, for scenes with long hair do remeshing for ```final_head.ply``` to properly deal with occlusions. You need to change the flag```render["mesh_path"] to  final_head_remeshed.ply path in ./configs/hair_strands_textured.yaml ```.

After this step in```./implicit-hair-data/data/SCENE_TYPE/CASE``` you would obtain ```hair_outer_remeshed.ply, final_head_remeshed.ply (optional)```.


#### Step 4. Extract scalp region for diffusion using the distance between hair sdf to scalp;

```bash
python ./preprocess_custom_data/cut_scalp.py --distance 0.07 --conf_path ./configs/SCENE_TYPE/neural_strands*.yaml  --case CASE --scene_type SCENE_TYPE --path_to_data ./implicit-hair-data/data 
```
Note, you could change the distance between scalp and hair sdf if obtained scalp mesh is too small or too big for current hairstyle.


After this step in ```./implicit-hair-data/data/SCENE_TYPE/CASE``` you would obtain ```cut_scalp_verts.pickle, scalp.obj, dif_mask.png```.


