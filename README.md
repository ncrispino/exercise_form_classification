# Exercise Form Classifier
Given a picture/video of an exercise, will classify it as done with conventionally good or bad form. This is different than some other projects as it aims not to only count repetitions, but to care about how the exercise is executed overall. Also, once given data, no heuristics like joint angles need to be used; the model should learn them itself. The neural network will be initially trained on large datasets for pose estimation and action recognition, then will be extended by training with self-labeled images for the supported exercises.

## Progress
- [x] Pose model finished training at around 80% on the PCKh measure (obtained by taking the mean across batches) on the MPII 2D Human Pose dataset using GCP. This is less than in the paper, so I should ideally return to this model.
- [] Train action recognition model (and implement validation score for it)
- [] Synthesize new dataset with exercises (have an idea using Google photos, but it doesn't seem like enough)
- [] Train on new data

## Supported Exercises
Currently, I am focusing on distinguishing straight vs. banana handstands. There are many other types of handstands (and exercises in general), but due to the cost of creating datasets, I will ignore them for now.

## Output
To-do.

## Requirements
To-do.

## Background
To-do.

## Methodology
As I care about how the joints are stacked, I want to use human pose estimation, then use the body positioning and overall image to classify the the action. To accomplish the initial task, I will implement "2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning" from scratch in PyTorch. I will be using the paper and the authors' Tensorflow code to do so. Like them, I will intially train the pose estimation part of the model with the MPII Human Pose Dataset and use the same preprocessing methods they do, as well as their utility functions which I put in the ```deephar_replication/deephar_utils``` folder. [Here is their repo](https://github.com/dluvizon/deephar). I should use the cvpr18 branch, as that's the paper I'm implementing, but I ended up basing most of my code off the master branch (for the TPAMI'20 paper). Note that the code for the datasets they used is mostly copied from their implementation, with slight modifications to work with PyTorch. See ```deephar_replication/data/datasets.md``` for more.

Note that I am [using Google docstrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings), which were pretty easy to do with the help of Github Copilot.

## Dataset
To-do.

I will initially start by classifying handstands as straight-line handstands. Once this works, I can extend the model to other exercises, perhaps comparing squat/deadlift/bench form to that of expert powerlifters.

To obtain the data, download from Google Images then use a program to label easily.

## Future Goals
### Showing Incorrect Form
A 'perfectly straight' handstand, perhaps according to averaging from the training data should be overlayed over the provided image to explain the differences. Or, perhaps I can transform the user inputted handstand into a straight-line handstand and place the images side-by-side, so that they could see where they're going wrong. To do this, I can use a General Adversarial Network (GAN), perhaps using the vector of results from the pose estimate as an input in addition to the image.

## Papers Implemented/Code Used
- [2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning](https://arxiv.org/pdf/1802.09232.pdf) combining action recognition and pose estimation using one network.

## General Information
### Combining Action Recognition and Pose Estimation
- [Coupled Action Recognition and Pose Estimation from
Multiple Views](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.5466&rep=rep1&type=pdf)
- [Notebook from Tensorflow that classifies Yoga Poses](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/pose_classification.ipynb#scrollTo=ydb-bd_UWXMq). This is very similar to what I want to be doing, though I won't use premade models and want to include additional information signifying how the final form is different from the target.
- [Pose Trainer: Correcting Exercise Posture using Pose Estimation
](https://arxiv.org/pdf/2006.11718.pdf) is a student paper combining pose estimation with exercise form using a geometric algorithm. However, they seem to specify a specific algorithm for each exercise; I want this to be automated. Also, I'll compare pose estimation results.
- [Google ML Kit Pose Detection API](https://developers.google.com/ml-kit/vision/pose-detection/classifying-poses)
- [Human Action Recognition using Detectron2 and LSTM](https://learnopencv.com/human-action-recognition-using-detectron2-and-lstm/#disqus_thread) is also similar to what I want to do.
### Focusing on Either Action Recognition Or Pose Estimation
- [DeepPose: Human Pose Estimation via Deep Neural Networks](https://arxiv.org/pdf/1312.4659.pdf) is generally important paper for pose estimation.
- [Understanding action recognition in still images](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w23/Girish_Understanding_Action_Recognition_in_Still_Images_CVPRW_2020_paper.pdf)
- [Pose Embeddings: A Deep Architecture for Learning to Match Human Poses](https://arxiv.org/abs/1507.00302)
- [Image similarity using Deep CNN and Curriculum Learning](https://arxiv.org/ftp/arxiv/papers/1709/1709.08761.pdf) for general image similarity. I don't plan on feeding pairs of images, though.
- [Exer-NN: CNN-Based Human Exercise Pose Classification](https://link.springer.com/chapter/10.1007/978-981-33-4367-2_34)
### GANs and Pose Estimation
- [Deformable GANs for Pose-based Human Image Generation](https://arxiv.org/abs/1801.00055) will likely be the main paper I will use to transform the current picture of the handstand to a straight one. To do this, I may have to aggregate the poses in the straight handstand images.
