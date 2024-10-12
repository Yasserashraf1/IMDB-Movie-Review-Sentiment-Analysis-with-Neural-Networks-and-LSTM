# IMDB-Movie-Review-Sentiment-Analysis-with-Neural-Networks-and-LSTM
This repository contains a deep learning project for Sentiment Analysis using the IMDB Movie Reviews Dataset. The project implements and compares a basic Neural Network (NN) and a Long Short-Term Memory (LSTM) model to classify movie reviews as either positive or negative. The models are built using TensorFlow/Keras and GloVe word embeddings for better text representation.

## Project Overview
The aim of this project is to develop a deep learning solution for classifying the sentiment of movie reviews from the IMDB dataset as positive or negative. We preprocess the reviews, convert them to numeric form using pre-trained GloVe embeddings, and then train both a Neural Network and an LSTM model. Finally, we compare their performances and use them for sentiment prediction on new reviews.

## Key Features:
* **Data Preprocessing:** Tokenization, stopword removal, padding.
* **Text Representation:** GloVe word embeddings.
* **Model Comparison:** Neural Network vs LSTM for sentiment analysis.
* **Evaluation Metrics:** Accuracy, loss.
* **New Sentence Prediction:** Predicting sentiment for new reviews.

## Dataset
We use the IMDB Movie Reviews Dataset which consists of 50,000 movie reviews categorized as either positive or negative. The dataset is split into:
Training set: 25,000 reviews
Test set: 25,000 reviews
The labels are:
1: Positive review
0: Negative review

## Source:
The dataset is available from the [IMDB Dataset on Kaggle]([url](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)).

## Data Preprocessing
The preprocessing steps include:
* **Text Cleaning:** Lowercasing, removing HTML tags, numbers, and special characters.
* **Tokenization:** Splitting the text into individual tokens (words).
* **Stopword Removal:** Removing common words that don't add significant meaning.
* **Padding:** Ensuring all reviews have the same length by padding shorter reviews.
The nltk library is used for stopword removal, and the Tokenizer class from Keras is used for tokenizing the text.

## Models
We implemented and trained two models:
* 1. Neural Network Model (NN)
  * Embedding Layer: Pre-trained GloVe embeddings to represent words.
  * Flatten Layer: Converts the 2D word embedding input to a 1D vector.
  * Dense Layer: Fully connected layer with ReLU activation for learning.
  * Output Layer: Sigmoid activation for binary classification (positive/negative).
* 2. LSTM Model
  * Embedding Layer: Same pre-trained GloVe embeddings.
  * LSTM Layer: A 100-unit LSTM with dropout and recurrent dropout to capture sequential patterns in text.
  * Output Layer: Sigmoid activation for binary classification.
  
## Embeddings
We used **GloVe** (Global Vectors for Word Representation) to represent words as dense vectors. Specifically, we used the 100-dimensional vectors from the GloVe 6B dataset. These embeddings were pre-trained on a large corpus of text and help improve the accuracy of our models by capturing semantic meanings of words.

* GloVe Embedding Details:
  * File: glove.6B.100d.txt
  * Source: [GloVe Project](https://nlp.stanford.edu/projects/glove/)

## Model Training
Both models were trained using:
* Loss Function: Binary Cross-Entropy.
* Optimizer: Adam.
* Metrics: Accuracy.
The training was performed over 10 epochs, and validation accuracy was monitored to evaluate the models’ performance.

## Hyperparameters:
* Max Words: 10,000 (maximum vocabulary size)
* Max Sequence Length: 100 (maximum length of each review)
* Embedding Dimension: 100 (dimension of GloVe vectors)

## Results and Evaluation
Both models were evaluated on the test set. The metrics used for evaluation are:
* Accuracy: Percentage of correctly predicted sentiments.
* Loss: Binary cross-entropy loss on the test set.

## How to Run
Prerequisites
* Python 3.x
* TensorFlow/Keras
* NLTK
* NumPy
* Matplotlib
* GloVe embeddings (glove.6B.100d.txt)

## Conclusion
This project demonstrates how to perform sentiment analysis on the IMDB movie reviews dataset using two different models, a simple Neural Network and an LSTM model. The use of GloVe embeddings significantly enhances the models' ability to understand word relationships and perform sentiment classification.
