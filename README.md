## DeepLiDARFlow
This is the source code of DeepLiDARFlow for dense scene flow estimation using monocular camera and sparse LiDAR.

## Requirements
The model is trained and tested on:  
- Windows 10 Pro and/or Ubuntu 18.04
- Python 3.7.8
    - gast==0.2.2 (to resolve incompatibilities between python version and tensorflow)
    - imageio==2.5.0
    - matplotlib==3.2.1
    - numpy==1.16.4
    - tensorflow-gpu==1.14
    - tqdm==4.31.1
- All required libraries can be installed by running `pip install -r requirements.txt`.

## Data Sets
- The original form of FlyingThings3D (FT3D) is used for pre-training and KITTI 2015 scene-flow is used for fine-tuning. All mappings have been done.
- For KITTI data set, downloading the de-warped disparities 1 to the view at time t+1 is required by running `sh download_preprocessed_kitti.sh`. 
Then, the folder should be placed in the KITTI data set (training) folder.

## Training and Fine-tuning
- Running `sh run_train.sh` will train the model from scratch. As a prerequisite, the correct path of FlyingThings3D data set should be provided for `--data_path` flag.
- Running `sh run_finetune.sh` will fine-tune the model. As a prerequisites, the correct path of KITTI data set should be provided for `--data_path` flag and 
  the pre-trained model path on FlyingThings3D should be provided for `--pretrained_model` flag.  
- During the training, best weights will be stored in the model directory and make a JSON file named `best_checkpoints`. If the best_checkpoints is available and 
  the flag`--best_checkpoint` is True, the training will continue with the best checkpoint.
 
## Maximum Epochs and Learning Rate Strategy
- For pre-training on FlyingThings3D, the maximum epochs number is defined with 600 epochs. The initial learning rate is 0.0001 and it's reduced to 0.00001 in case of over-fitting.
- For fine-tuning on KITTI, the model needs to be fine-tuned for 200. First 100 epochs are trained using 0.0001 learning rate, then it's reduced to 0.00001 for the rest of epochs.
- If the training or fine-tuning will be continued after changing learning rates or maximum epochs number, the `--best_checkpoint` flag should be activated to call the best weights for continuing the training.  

## Evaluation and Inference
- Running `sh run_evaluation.sh` will evaluate the whole frames in the test split of FlyingThings3D and the mapped frames of KITTI in 'KITT_test.txt'.  
    - The number of LiDAR samples can be estimated by setting the `--samples` flag [default: 5000]. 
    - The data set type (either 'KITTI' or 'FT3D' should be provided for the flag `--dataset`.
    - The path of the data set should be given for `--data_path`.
    - The model path should be given for `--model_path` [default: 'model/DeepLiDARFlow'].      
      The software can take the suitable model based on data set type `--dataset`.
- Running `sh run_inference.sh` can evaluate the model on two selected frames inside 'images' folder (one is KITTI example and another  is FT3D). 
    - 5000 of LiDAR points are considered with these frames.
    - The example flag `--ex` can be specified either by '1' for KITTI frame or '2' for FT3D frame. 
    - The model path should be given for `--model_path` [default: 'model/DeepLiDARFlow'] and the software can take the suitable model based on image resolution.

## Model Weights
- The execution of `download_models.sh` can download the weights for both data sets (FT3D/KITTI) inside the 'model' folder:
```
	model/DeepLiDARFlow-FT3D
	model/DeepLiDARFlow-KITTI
```

## Citation
If you find the code or the paper useful, consider citing us:
```
@inproceedings{DeepLiDARFlow2020,
  title={{DeepLiDARFlow: A Deep Learning Architecture For Scene Flow Estimation Using Monocular Camera and Sparse LiDAR}},
  author={Rishav and Battrawy, Ramy and Schuster, Ren{\'e} and Wasenm{\"u}ller, Oliver and Stricker, Didier},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020},
}
```