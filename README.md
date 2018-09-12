
# Cascade Networks For Single Object Detection

## Introduction

This is a project for single object detection. Note that in object single object detection, one image can have multiple objects, but all the objects in the dataset must belong to the same class. In this project,  [Multi-task Cascaded Neural Networks (MTCNN)](https://arxiv.org/abs/1604.02878) is used. This project contains python scripts of MTCNN face detector, training testing scripts, dataset examples, and other codes. This implementation is heavily based on [this project](https://github.com/AITTSMD/MTCNN-Tensorflow) .

## Some Examples

![Alt text](https://github.com/Ao-Lee/SingleObjectDetection/raw/master/test/results/2.png)

![Alt text](https://github.com/Ao-Lee/SingleObjectDetection/raw/master/test/results/1.png)

## How to use the pre-trained model
open config\cfg.py, and change the 'path_code' variable to \<project_path\> where you extract this code. Then, there are two testing scripts which you can test pretrained models. Datasets used for testing is self contained in this project.
* run test\one_image_test.py to check the ground truth bounding boxes and prediction bounding boxes. 
* To quantify the performance, [average precision](https://github.com/Ao-Lee/AveragePrecision) is used in this case, run the script test\test_on_ap.py for more details

## How to train the model 
#### about the  data
example training dataset is self contained. It is located in <\project_path\>\test\datasets\TrainingSets. 
#### training
1. open open config\cfg.py, change the 'path_code' variable to \<project_path\>. 
2. in cfg.py, change the 'debug' variable to 'True'. 
3. in cfg.py, change the 'path_output_files' variable to an empty folder.
4. run the script script.py to train the network

Note that this is just an simple example. This dataset is too small to train any model. However, you can still check what the training dataset looks like. If there is no error reports (hopefully), it is time to train the network with your own data.

## How to train the model with my own data
#### prepare images
1. put every image in a single folder, note that this folder should not contain any sub-folders.
2. in cfg.py, change the variable 'path_detection_imgs' to the image folder you prepared.
#### prepare ground truth labels
1. all the labels should be in a txt file. Each line represents a file corresponding to an image in the image folder. For example:
`01.jpg 1624 392 1919 987 733 7 1266 818`
`02.jpg 371 515 666 735`
`03.jpg 23 521 333 694 721 461 928 662 578 505 842 713`
01.jpg has two bounding boxes, the first one is [1624, 392, 1919, 987], the second one is [733, 7, 1266, 818], each box has the form [left, top, right, bot]
2. in cfg.py, change the variable 'path_detection_labels' to the txt file prepared.
3. in cfg.py change the variable 'path_output_files' to empty folder. Warning, this folder requires large disk space. (depending on the size of your data)
4. in cfg.py, change the variable 'debug' to False.
5. run script.py to train the network with your own data.
#### train the model on CrowdAI Dataset
what if I don't have any available dataset, and I want to train the network with a complete dataset?

Well, you can try some vehicle detection dataset, for example,  [CrowdAI](http://crowdai.com/) dataset and [Autti](http://autti.co/) dataset. It is convenient to check the details and download the data from [this](https://github.com/udacity/self-driving-car/tree/master/annotations) project.

Note that the labelling format in those datasets are not directly supported, use \Preprocess_Vehicle\preprocess_crowdAI.py to convert the labels.

## Model Optimization
In most cases and most projects, model optimization requires tons of work with trial and error. There are lots of factors that affect model performance: data pre-processing, network structure, learning rate, weights initialization,..., you name it. The annoying part is that if you have a new idea, you need to retrain the network to see the results. 

This part only focus on model-free hyper parameter optimization. Since   [(MTCNN)](https://arxiv.org/abs/1604.02878) is a network framework that contains several small networks, there are some parameters that control how theses small networks are used in detection. These parameters are independent to the training process, so there is an easy to tune these parameters even after training.

See test\optimise_hyper.py for more details. It is recommenced that this script is used on testing dataset rather than training dataset.


