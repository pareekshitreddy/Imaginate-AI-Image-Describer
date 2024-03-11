# Imaginate: AI Image Describer

This project aims to generate natural language captions for images using deep learning models. It explores different architectures and techniques such as CNN-LSTM, CNN-Transformer, and BERT embeddings.

## Requirements and Dependencies
- Python 3.7 or higher
- TensorFlow 2.0 or higher
- Keras
- Numpy
- Matplotlib
- NLTK
- Gensim
- HuggingFace Transformers
- Flickr8k and Flickr30k datasets

## Installation and Usage
1. Clone this repository: `git clone https://github.com/your-username/image-captioning.git`
2. Install the required packages
3. Download the datasets from [here](https://www.kaggle.com/datasets/adityajn105/flickr8k) and [here](https://www.kaggle.com/datasets/adityajn105/flickr30k) and place them in the `data` folder


## Results and Evaluation
The models were evaluated using the BLEU scores on the Flickr8k and Flickr30k datasets. The following table summarizes the results:

| Model           | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-----------------|--------|--------|--------|--------|
| CNN-RNN         | 0.452  | 0.481  | 0.508  | 0.440  |
| CNN-Transformer | 0.389  | 0.551  | 0.633  | 0.655  |
| CNN-BERT        | 0.402  | 0.526  | 0.782  | 0.655  |

Some examples of generated captions are shown below:

- CNN-RNN: A group of people sitting on a couch with a laptop
- CNN-Transformer: A group of friends using a laptop on a sofa
- CNN-BERT: A group of people sitting on a couch and looking at a laptop
- CNN-RNN: A man riding a bike on a dirt road
- CNN-Transformer: A man riding a mountain bike on a trail
- CNN-BERT: A man riding a bicycle on a dirt path

## Credits and References
This project was inspired by the following papers and resources:
- [Image Captioning Based on Deep Neural Networks] by Liu et al.
- [A Thorough Review of Models, Evaluation Metrics, and Datasets on Image Captioning] by Luo et al.
- [Unified Vision-Language Pre-Training for Image Captioning and VQA] by Zhou et al.
- [Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning] by Sharma et al.
- [Attention Is All You Need] by Vaswani et al.
- [RUBERT: A Bilingual Roman Urdu BERT Using Cross Lingual Transfer Learning] by Khalid et al.
- [Flickr Image Datasets] by Hsankesara
- [GloVe: Global Vectors for Word Representation] by Pennington et al.
