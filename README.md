# Handstand Form Classifier
Given a picture of a handstand from the side, will classify the form as straight or not, then highlight the descrepancies with an image (and possibly text in the future). Note that at this time, I am mainly concerned with distinguishing straight vs. banana handstands. There are many other types of handstands, but due to the cost of creating datasets, I will ignore them for now. This model could easily be extended to one that classifies the form for these different types.

## Output
To-do.

## Requirements
To-do.

## Background
To-do.

## Methodology
As I care about how the joints are stacked, I want to use human pose estimation, then use the body positioning to classify the image. A 'perfectly straight' handstand, perhaps according to averaging from the training data, will be overlayed over the provided image to explain the differences. Right now, I'm thinking of using a pre-trained model for pose estimation, then using a model I replicated for action recognition. OR, I could do both in the same model, but it would require a lot of GPU power. Then, to show what they did wrong, perhaps I can transform the user inputted handstand into a straight-line handstand and place the images side-by-side, so that they could see where they're going wrong. To do this, I think I will use a General Adversarial Network (GAN), perhaps using the vector of results from the pose estimate as an input in addition to the image.

## Dataset
To-do.

## Sources
### Papers Implemented
- [2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning](https://arxiv.org/pdf/1802.09232.pdf) combining action recognition and pose estimation using one network. **I'm currently in the process of implementing**.

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
