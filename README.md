# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.


# Writeup
## Project overview

In this project, you'll find code to create a convolutional neural network to detect and classify objects using data from Waymo. For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/). The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records.

## Dataset

### Dataset analysis

The data provided in the course is organized in `/home/workspace/data/waymo/training_and_validation` where we can initially see 97 tfrecord files. 
Here we can observe random extraction of image data, included objects and their bounding boxes.  

![output0](./images/output0.png)
![output1](./images/output1.png)
![output2](./images/output2.png)
![output3](./images/output3.png)
![output4](./images/output4.png)
![output5](./images/output5.png)
![output6](./images/output6.png)
![output7](./images/output7.png)
![output8](./images/output8.png)

As we see, we can see quite many cars detected compared to pedestrians.  In this small extraction, we can't see bikes which indicates that we might have inbalanced object class distribution.  
On top of the class variation, we can see images variability including several weather conditions and several brightness/contrast patterns.  

Indeed, here is an actual aggregation of the classes in the dataset.  
- **car**: 540043 (76%)
- **pedestrian**: 161468 (22%)
- **cyclist**: 4182 (less than 1%)

Next, we can analyse how many objects are included in a image. We can focus on the most common objects of cars.  Here is a histogram of the number of cars in an image.  

![cars_in_image](./images/cars_in_image.png)

We can see there's not much of specific pattern but most of the values lie under 40.  

### Cross validation

The goal of our ML algorithm is to be deployed in a production environment. But before we can deploy such algorithms, we need to be sure that it will perform well in any environments it will encounter. In other words, we want to evaluate the generalization ability of our model.  
Out of many available available cross validation methods exist, I adoped a simple training, test, validation split because LOO (Leave One Out) or k-fold cross validation won't fit for out case due to computational expense of CNN.  

In the `split` function found in `create_split.py`, you can see training, test and validation split ratio as
- training: 75%
- test: 15%
- validation: 10%

The results of this split is found in my workspace  
- /home/workspace/data/train: 72 tfrecords
- /home/workspace/data/train: 15 tfrecords
- /home/workspace/data/train: 10 tfrecords

## Training

As given in the course, I have run a couple of experiemnts where `/home/workspace/training` is the first experiment based on `pipeline_new.config` obtained by 
```
python edit_config.py \
    --train_dir /home/workspace/data/train/ \
    --eval_dir /home/workspace/data/val/ \
    --batch_size 4 \
    --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 \
    --label_map label_map.pbtxt
```

and `/home/workspace/training2` contains records and config based on my edits after observing the first trial which I'll address in detail in the `Improve on the reference` chapter.  

### Reference experiment

Here you see the results of the first trial given by Tensorboard.  

![training](./images/training.png)

As the course describes as "Most likely, this initial experiment did not yield optimal results", even though regularization loss decreases after a heap, classification loss didn't decrease at all.  

### Improve on the reference

After I checked data argumentation options available in [the provided proto file](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) particularly
```
message PreprocessingStep {
  oneof preprocessing_step {
    NormalizeImage normalize_image = 1;
    RandomHorizontalFlip random_horizontal_flip = 2;
    RandomPixelValueScale random_pixel_value_scale = 3;
    RandomImageScale random_image_scale = 4;
    RandomRGBtoGray random_rgb_to_gray = 5;
    RandomAdjustBrightness random_adjust_brightness = 6;
    RandomAdjustContrast random_adjust_contrast = 7;
    RandomAdjustHue random_adjust_hue = 8;
    RandomAdjustSaturation random_adjust_saturation = 9;
    RandomDistortColor random_distort_color = 10;
    RandomJitterBoxes random_jitter_boxes = 11;
    RandomCropImage random_crop_image = 12;
    RandomPadImage random_pad_image = 13;
    RandomCropPadImage random_crop_pad_image = 14;
    RandomCropToAspectRatio random_crop_to_aspect_ratio = 15;
    RandomBlackPatches random_black_patches = 16;
    RandomResizeMethod random_resize_method = 17;
    ScaleBoxesToPixelCoordinates scale_boxes_to_pixel_coordinates = 18;
    ResizeImage resize_image = 19;
    SubtractChannelMean subtract_channel_mean = 20;
    SSDRandomCrop ssd_random_crop = 21;
    SSDRandomCropPad ssd_random_crop_pad = 22;
    SSDRandomCropFixedAspectRatio ssd_random_crop_fixed_aspect_ratio = 23;
    SSDRandomCropPadFixedAspectRatio ssd_random_crop_pad_fixed_aspect_ratio =
        24;
    RandomVerticalFlip random_vertical_flip = 25;
    RandomRotation90 random_rotation90 = 26;
    RGBtoGray rgb_to_gray = 27;
    ConvertClassLogitsToSoftmax convert_class_logits_to_softmax = 28;
    RandomAbsolutePadImage random_absolute_pad_image = 29;
    RandomSelfConcatImage random_self_concat_image = 30;
    AutoAugmentImage autoaugment_image = 31;
    DropLabelProbabilistically drop_label_probabilistically = 32;
    RemapLabels remap_labels = 33;
    RandomJpegQuality random_jpeg_quality = 34;
    RandomDownscaleToTargetPixels random_downscale_to_target_pixels = 35;
    RandomPatchGaussian random_patch_gaussian = 36;
    RandomSquareCropByScale random_square_crop_by_scale = 37;
    RandomScaleCropAndPadToSquare random_scale_crop_and_pad_to_square = 38;
    AdjustGamma adjust_gamma = 39;
  }
}
```

,
given our dataset contains dart and/or blue images  
![output3](./images/output3.png)
![output4](./images/output4.png)

I decided to include more gray scale and adjust contrast.  
The edited `pipeline_new.config` is found in `/home/workspace/training2/reference/pipeline_new.config` particularll the additional part is

```
  data_augmentation_options {
    random_rgb_to_gray {
    probability: 0.3
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
    min_delta: 0.5
    max_delta: 1.0
    }
  }
```

After the training and evaluation steps, we can observe the results again using Tensorboard


![training2](./images/training2.png)

Now it's obvious that classification loss decreases indicating out training process works better.

