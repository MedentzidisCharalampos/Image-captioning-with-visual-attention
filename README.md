# Image-captioning-with-visual-attention
Given an image like the example below, our goal is to generate a caption such as "a surfer riding on a wave".  

<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Image-captioning-with-visual-attention/blob/main/image_caption.jpg" />
</p>  
<p align="center">  Figure 1.Image the caption is to be generated. <p>

 We use an attention-based model, which enables us to see what parts of the image the model focuses on as it generates a caption.
 
<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Image-captioning-with-visual-attention/blob/main/prediction_caption.png" />
</p>  
<p align="center">  Figure 1. Prediction of the Caption. <p>
  
# Download the dataset

We will use the MS-COCO dataset (https://cocodataset.org/#home) to train our model. 
The dataset contains over 82,000 images, each of which has at least 5 different caption annotations. 
To speed up training for this tutorial, we use a subset of 30,000 captions and their corresponding images to train our model.
Choosing to use more data would result in improved captioning quality.

# Preprocess the images 

We use InceptionV3 (which is pretrained on Imagenet) to classify each image. We extract features from the last convolutional layer.

First, we convert the images into InceptionV3's expected format by:

1. Resizing the image to 299px by 299px
2. Preprocess the images to normalize the image so that it contains pixels in the range of -1 to 1, which matches the format of the images used to train InceptionV3.


Load the pretrained Imagenet weights:

We create a tf.keras model where the output layer is the last convolutional layer in the InceptionV3 architecture. The shape of the output of this layer is 8x8x2048.

1. We forward each image through the network and store the resulting vector in a dictionary (image_name --> feature_vector).
2. After all the images are passed through the network, you pickle the dictionary and save it to disk.

Caching the features extracted from InceptionV3:

We pre-process each image with InceptionV3 and cache the output to disk. Caching the output in RAM would be faster but also memory intensive, requiring 8 * 8 * 2048 floats per image. 

# Preprocess and tokenize the captions

1. We tokenize the captions (for example, by splitting on spaces). This gives us a vocabulary of all of the unique words in the data (for example, "surfing", "football", and so on).
2. We limit the vocabulary size to the top 5,000 words (to save memory). We replace all other words with the token "UNK" (unknown).
3. We create word-to-index and index-to-word mappings.
4. We pad all sequences to be the same length as the longest one.

# Model

The model architecture is inspired by the Show, Attend and Tell paper (https://arxiv.org/pdf/1502.03044.pdf).

1. Extract the features from the lower convolutional layer of InceptionV3 giving us a vector of shape (8, 8, 2048).
2. Squash that to a shape of (64, 2048).
3. This vector is then passed through the CNN Encoder (which consists of a single Fully connected layer).
4. The RNN (here GRU) attends over the image to predict the next word.

# Training

1. Extract the features and pass those features through the encoder.
2. The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
3. The decoder returns the predictions and the decoder hidden state.
3. The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
4. Use teacher forcing to decide the next input to the decoder.
5. Teacher forcing is the technique where the target word is passed as the next input to the decoder.
6. The final step is to calculate the gradients and apply it to the optimizer and backpropagate.

# Caption

1. The evaluate function is similar to the training loop, except you don't use teacher forcing here. The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.
2. Stop predicting when the model predicts the end token.
3. And store the attention weights for every time step.

# The Output of the Example

<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Image-captioning-with-visual-attention/blob/main/output_caption.png" />
</p>  
<p align="center">  Figure 3. The predicted caption. <p>
