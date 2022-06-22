# Image-Captioning

Large number of images are generated each day through a wide variety of mediums
such as internet, news articles and advertisements. Most of these images do not have a caption or
description yet we humans can recognize them with minimum effort. But, this is not the same for
a machine. Image captioning is used extensively for Content-Based Image Retrieval (CBIR) and
applied in many fields such as education, military, bio-medicine and web-scraping. Also, it is used in
social media platforms for identifying where a particular picture is taken, what kind of a picture it is
and describe any activity that is shown in the picture which can be further used for recommending
content to the user.
Image Captioning is the process of generating description of an image from the objects and actions in
the image. Image captioning is a multi-modal problem that has drawn extensive attention in both the
natural language processing and computer vision community.
Understanding an image mainly depends on obtaining image features. The techniques used for
this purpose can be broadly divided into two categories: (1) Traditional machine learning based
techniques and (2) Deep machine learning based techniques. Traditional machine based techniques
involves a combination of hand crafted features like Local Binary Patterns and Histogram of Oriented
Gradients. Since hand crafted features are task specific, extracting features from a large and diverse
set of data is not feasible. Moreover, real world data such as images and video are complex and
have different semantic interpretations. But in deep machine learning based techniques learn the
features automatically and can handle a wide variety of data-sets like images and videos. Consider,
Convolutional Neural Networks (CNN) that are widely used for feature learning, and a classifier such
as Softmax is used for classification.
In this project, we will implement baseline model using CNN as the encoder and LSTM as a decoder
and evaluate it using BLEU[6] score as a metric.
As an intermediate model, we will implement a CNN + Transformer based model using Attention
Mechanism for Caption Generation and evaluate it using BLEU[6] score as a metric.
For the final model, we investigate the effect of pre-trained embeddings on the task of image
captioning by BERT context vectors [12] to enhance the models performance and reduce training
time.
The task is divided into two modules, namely Image base and Language based module. Image based
module extracts the features and nuances out of the image and the Language based module translates
the features and objects from the image to a natural sentence. For the image based module to encode
the image we used CNN and for the decoder we used Pre-Trained BERT along with LSTM.
