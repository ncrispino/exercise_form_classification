# Handstand Form Classifier
Given a picture of a handstand from the side, will classify the form as straight or not, then highlight the descrepancies with an image (and possibly text in the future). Note that at this time, I am mainly concerned with distinguishing straight vs. banana handstands. There are many other types of handstands, but due to the cost of creating datasets, I will ignore them for now. This model could easily be extended to one that classifies the form for these different types.

## Background
To-do.

## Methodology
As I care about how the joints are stacked, I want to use human pose estimation, then use the body positioning to classify the image. A 'perfectly straight' handstand, according to averaging from the training data, will be overlayed over the provided image to explain the differences.

## Dataset
To-do.

## Sources
- [Notebook from Tensorflow that classifies Yoga Poses](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/pose_classification.ipynb#scrollTo=ydb-bd_UWXMq). This is very similar to what I want to be doing, though I won't use premade models and want to include additional information signifying how the final form is different from the target.
- [Pose Embeddings: A Deep Architecture for Learning to Match Human Poses](https://arxiv.org/abs/1507.00302)
