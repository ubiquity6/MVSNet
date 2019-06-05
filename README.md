# MVSNet

## Training on ml-engine

* Clone the repo
* Install gcloud command line tools

```export JOB_NAME=your-unique-job-name```

```
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir gs://mvs-training-mlengine/$JOB_NAME \
    --module-name mvsnet.train \
    --package-path /Users/chrisheinrich/ml/MVSNet/mvsnet \
    --region us-central1 \
    --runtime-version 1.13 \
    --config /Users/chrisheinrich/ml/MVSNet/machines/1p100.yaml \
    -- \
    --train_data_root gs://mvs-training-mlengine/dtu_7scenes_rgbd_scenes11_fixed/ \
    --log_dir gs://mvs-training-mlengine/$JOB_NAME/logs/ \
    --model_dir gs://mvs-training-mlengine/$JOB_NAME/models \
    --epoch 1 \
    --num_gpus 1 
```

This command would train MVSNet for 1 epochs on a dataset of consisting of DTU, scenes11, RGBD and 7 Scenes on a p100 GPU.


## Inference on Atlas data

You can convert Atlas data into the format read by MVSNet for inference using the following command:

```
bzr ubq/ai/tools:map-to-mvs-training -- --map-id <atlas-map-id> --stack <stack> --data-dir <dir-to-deposit-data>
```

Which would download a map from Atlas and then conver to the right format. Then you can do inference on that data by running this command from the root of the mvsnet repo

```
python -m mvsnet.test --dense_folder <atlas-data-in-densify-train-format> --ckpt_step <ckpt-of-saved-model> --model_dir <dir-where-saved-model-is>
```

Note that `--model_dir` can be a google storage bucket where models were saved during training with ml-engine, so you could run:

```
python -m mvsnet.test --dense_folder <atlas-data-in-densify-train-format> --ckpt_step 35000 --model_dir gs://mvs-training-mlengine/dtu_scan_104_epochs_200_lr_0025_viewnum_4/models/ 
```

To use one of our trained models trained for 35000 steps


## Inference plus point cloud fusion

The library we use to fuse MVSNet depth maps (fusibile - https://github.com/kysucix/fusibile ), requires a GPU to run, so if you want to use it you will need to connect to a GPU box. One that you can use, which already has fusibile installed, is a google cloud box named deepmvs-vm. To connect to this you will need the gcloud command line tools. Next, you can run:

```
gcloud compute instances start deepmvs-vm  # starts the instance
gcloud compute ssh deepmvs-vm --zone us-west1-b # connects to the instance
cd MVSNet
nohup python -m scripts.test_and_fuse --ckpt_step 1350000 --model_dir ./model --prob_threshold 0.1 --disp_threshold 0.2 --num_consistent 3 --test_folder_root /home/chrisheinrich/data/atlas2/ --no_test &
```

This woud run our `test_and_fuse` script which performs inference on all of the Atlas test data we have on that machine (you could add more), fuses the depth maps, and then uploads the results to sketchfab. At the end of all this you can run:

```
cat nohup.out
```

to view the output, the last few lines of which will contain the URLs of the uploaded PLYs on sketchfab. The prob_threshold and disp_threshold are arguments passed to fusibile for point cloud fusion. Heuristically speaking:
* Higher prob_threshold leads to higher precision and lower recall
* Lower disp_threshold leads to higher precision and lower recall
* Higher num_consistent leads to higher precision and lower recall

or vice versa.





## NOTE -- old documentation from original branch included below. Some of this may now be outdated

## About
[MVSNet](https://arxiv.org/abs/1804.02505) is a deep learning architecture for depth map inference from unstructured multi-view images, and [R-MVSNet](https://arxiv.org/abs/1902.10556) is its extension for scalable learning-based MVS reconstruction. If you find this project useful for your research, please cite:
```
@article{yao2018mvsnet,
  title={MVSNet: Depth Inference for Unstructured Multi-view Stereo},
  author={Yao, Yao and Luo, Zixin and Li, Shiwei and Fang, Tian and Quan, Long},
  journal={European Conference on Computer Vision (ECCV)},
  year={2018}
}
```
```
@article{yao2019recurrent,
  title={Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference},
  author={Yao, Yao and Luo, Zixin and Li, Shiwei and Shen, Tianwei and Fang, Tian and Quan, Long},
  journal={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## How to Use

### Installation

* Check out the source code ```git clone https://github.com/YoYo000/MVSNet```
* Install cuda 9.0, cudnn 7.0 and python 2.7
* Install Tensorflow and other dependencies by ```sudo pip install -r requirements.txt```

### Training

* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) (Fixed training cameras, Sep. 19), and upzip it as the ``MVS_TRANING`` folder
* Enter the ``MVSNet/mvsnet`` folder, in ``train.py``, set ``dtu_data_root`` to your ``MVS_TRANING`` path
* Create a log folder and a model folder in wherever you like to save the training outputs. Set the ``log_dir`` and ``save_dir`` in ``train.py`` correspondingly
* Train MVSNet (GTX1080Ti): 
``python train.py --regularization 'GRU'`` 
* Train R-MVSNet (GTX1080Ti):
``python train.py --regularization '3DCNNs'``

### Testing

* Download the test data for [scan9](https://drive.google.com/file/d/17ZoojQSubtzQhLCWXjxDLznF2vbKz81E/view?usp=sharing) and unzip it as the ``TEST_DATA_FOLDER`` folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file
* Download the pre-trained MVSNet and R-MVSNet [models](https://drive.google.com/file/d/1h40Rq8ou5XLGFFSXTFBvrLla7-RMz73n/view) and upzip the file as ``MODEL_FOLDER``
* Enter the ``MVSNet/mvsnet`` folder, in ``test.py``, set ``model_dir`` to ``MODEL_FOLDER``
* Run MVSNet (GTX1080Ti): 
``python test.py --dense_folder TEST_DATA_FOLDER  --regularization '3DCNNs' --width 1152 --height 864 --max_d 192 --interval_scale 1.06``
* Run R-MVSNet (GTX1080Ti): 
``python test.py --dense_folder TEST_DATA_FOLDER  --regularization 'GRU' --width 1600 --height 1200 --max_d 256 --interval_scale 0.8``
* Inspect the .pfm format outputs in ``TEST_DATA_FOLDER/depths_mvsnet`` using ``python visualize.py .pfm``. For example the depth map and probability map for image `00000012` should be something like:

<img src="doc/image.png" width="250">   | <img src="doc/depth_example.png" width="250"> |  <img src="doc/probability_example.png" width="250">
:---------------------------------------:|:---------------------------------------:|:---------------------------------------:
reference image                          |depth map                                |  probability map 


### Post-Processing

R/MVSNet itself only produces per-view depth maps. To generate the 3D point cloud, we need to apply depth map filter/fusion for post-processing. As our implementation of this part is depended on the [Altizure](https://www.altizure.com/) internal library, currently we could not provide the corresponding code. Fortunately, depth map filter/fusion is a general step in MVS reconstruction, and there are similar implementations in other open-source MVS algorithms. We provide the script ``depthfusion.py`` to utilize [fusibile](https://github.com/kysucix/fusibile) for post-processing (thank Silvano Galliani for the excellent code!). 

To run the post-processing: 
* Check out the modified version fusibile ```git clone https://github.com/YoYo000/fusibile```
* Install fusibile by ```cmake .``` and ```make```, which will generate the executable at ``FUSIBILE_EXE_PATH``
* Run post-processing (--prob_threshold 0.8 if using 3DCNNs):
``python depthfusion.py --dense_folder TEST_DATA_FOLDER --fusibile_exe_path FUSIBILE_EXE_PATH --prob_threshold 0.3``
* The final point cloud is stored in `TEST_DATA_FOLDER/points_mvsnet/consistencyCheck-TIME/final3d_model.ply`.

We observe that the point cloud output of ``depthfusion.py`` is very similar to our own implementation. For detailed differences, please refer to [MVSNet paper](https://arxiv.org/abs/1804.02505) and [Galliani's paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Galliani_Massively_Parallel_Multiview_ICCV_2015_paper.pdf). The point cloud for `scan9` should look like:


<img src="doc/fused_point_cloud.png" width="375">   | <img src="doc/gt_point_cloud.png" width="375"> 
:--------------------------------------------------:|:----------------------------------------------:
point cloud result                          |ground truth point cloud


### Reproduce Benchmarking Results

The following steps are required to reproduce the point cloud results:

* Generate R/MVSNet inputs from the SfM outputs, you can use our preprocessed inputs for [DTU](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_), [Tanks and Temples](https://drive.google.com/open?id=1YArOJaX9WVLJh4757uE8AEREYkgszrCo) and [ETH3D](https://drive.google.com/open?id=1hGft7rEFnoSrnTjY_N6vp5j1QBsGcnBB) datasets (provided)
* Run R/MVSNet test script to generate depth maps for all views (provided)
* Apply variational depth map refinement for all views (optional, not provided)
* Apply depth map filter and fusion to generate the point cloud results (partially provided via fusibile)

R-MVSNet point cloud results with full post-processing are also provided: [DTU evaluation point clouds](https://drive.google.com/open?id=1L0sQjIVYu2hYjwpwbWSN8k42QhkQDjbQ) 

## File Formats

Each project folder should contain the following
```
.                          
├── images                 
│   ├── 00000000.jpg       
│   ├── 00000001.jpg       
│   └── ...                
├── cams                   
│   ├── 00000000_cam.txt   
│   ├── 00000001_cam.txt   
│   └── ...                
└── pair.txt               
```
If you want to apply R/MVSNet to your own data, please structure your data into such a folder.

### Image Files
All image files are stored in the `images` folder. We index each image using an 8 digit number starting from `00000000`. The following camera and output files use the same indexes as well. 

### Camera Files
The camera parameter of one image is stored in a ``cam.txt`` file. The text file contains the camera extrinsic `E = [R|t]`, intrinsic `K` and the depth range:
```
extrinsic
E00 E01 E02 E03
E10 E11 E12 E13
E20 E21 E22 E23
E30 E31 E32 E33

intrinsic
K00 K01 K02
K10 K11 K12
K20 K21 K22

DEPTH_MIN DEPTH_INTERVAL (DEPTH_NUM DEPTH_MAX) 
```
Note that the depth range and depth resolution are determined by the minimum depth `DEPTH_MIN`, the interval between two depth samples `DEPTH_INTERVAL`, and also the depth sample number `DEPTH_NUM` (or `max_d` in the training/testing scripts if `DEPTH_NUM` is not provided). We also left the `interval_scale` for controlling the depth resolution. The maximum depth is then computed as:
```
DEPTH_MAX = DEPTH_MIN + (interval_scale * DEPTH_INTERVAL) * (max_d - 1)
``` 

### View Selection File
We store the view selection result in the `pair.txt`. For each reference image, we calculate its view selection scores with each of the other views, and store the 10 best views in the pair.txt file:
```
TOTAL_IMAGE_NUM
IMAGE_ID0                       # index of reference image 0 
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 0 
IMAGE_ID1                       # index of reference image 1
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 1 
...
``` 


### Output Format
The ``test.py`` script will create a `depths_mvsnet` folder to store the running results, including the depth maps, probability maps, scaled/cropped images and the corresponding cameras. The depth and probability maps are stored in `.pfm` format. We provide the python IO for pfm files in the `preprocess.py` script, and for the c++ IO, we refer users to the [Cimg](http://cimg.eu/) library. To inspect the pfm format results, you can simply type `python visualize.py .pfm`. 

## Todo

* Validation script
* View selection from Altizure/COLMAP/OpenMVG SfM output 
* Depth sample selection from Altizure/COLMAP/OpenMVG SfM output 

## Changelog

### 2019 Feb 28 
* Use `tf.contrib.image.transform` for differentiable homography warping. Reconstruction is now x2 faster!

### 2019 March 1 
* Implement R-MVSNet and GRU regularization
* Network change: enable scale and center in batch normalization
* Network change: replace UniNet with 2D UNet 
* Network change: use group normalization in R-MVSNet

### 2019 March 7
* MVSNet / R-MVSNet and training / testing scripts
* MVSNet and R-MVSNet models (trained for 100000 iterations)

### 2019 March 11
* Add "Reproduce Benchmarking Results" section

### 2019 March 14
* Add R-MVSNet point clouds of DTU evaluation set




