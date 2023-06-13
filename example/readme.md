You could download one of [monocular scenes](https://drive.google.com/file/d/1CADXQfC2IgxmFLwcLrm4G3ilWpW1g_PA/view?usp=sharing) with checkpoints.

In ```NeuralHaircut``` folder do the following:

```bash
gdown 1CADXQfC2IgxmFLwcLrm4G3ilWpW1g_PA && unzip implicit-hair-data.zip && rm *.zip
```

##### Launch first stage:

```bash
python run_geometry_reconstruction.py --case person_0 --conf ./configs/example_config/neural_strands-monocular.yaml --exp_name first_stage_person_0
```

Results saved in folder ```./exps_first_stage/first_stage_person_0/```

##### Launch second stage:

```bash
python run_strands_optimization.py --case person_0 --scene_type monocular --conf ./configs/example_config/neural_strands-monocular.yaml  --hair_conf ./configs/example_config/hair_strands_textured.yaml --exp_name second_stage_person_0
```

Results saved in folder ```./exps_second_stage/second_stage_person_0/```


##### To see the training curves run:

```bash
tensorboard --logdir ./exps_{first|second}_stage/{first|second}_stage_person_0/
```