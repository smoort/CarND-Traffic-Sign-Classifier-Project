# **Traffic Sign Recognition Project Writeup**

### This is a write up on the German Traffic sign recognition project

---

### **Goals of the Traffic Sign Recognition Project**

* Load the input data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
* Optionally, visualize the Neural Network's State with Test Images


[//]: # (Image References)

[image1]: ./writeupImgs/TrainingDatasetSpread.png "Training Data"
[image2]: ./writeupImgs/Classes.png "Class verfication"
[image3]: ./writeupImgs/AugDatasetSpread.png "Augumented Data"
[image4]: ./writeupImgs/Speed20-1.jpg "Original image"
[image5]: ./writeupImgs/Speed20-2.jpg "Jittered image"
[image6]: ./writeupImgs/Speed20-3.jpg "Jittered image"
[image7]: ./writeupImgs/Speed20-4.jpg "Jittered image"
[image8]: ./writeupImgs/Speed20-5.jpg "Jittered image"
[image9]: ./writeupImgs/01SpeedLimit30.jpg "Speed Limit 30" 
[image10]: ./writeupImgs/11RighOfWay.jpg "Right of Way"
[image11]: ./writeupImgs/17NoEntry.jpg "No Entry"
[image12]: ./writeupImgs/28Childcrossing.jpg "Child Crossing"
[image13]: ./writeupImgs/34LeftAhead.jpg "Left Ahead"
[image14]: ./writeupImgs/LeftAheadLarge.jpg "Left Ahead Large"
[image15]: ./writeupImgs/visualize_cnn.png "Conv Layer"
[image16]: ./writeupImgs/Predictions.png "Bar graph showing predictions"
[image17]: ./writeupImgs/AccuracyGraph.png "Line graph showing training loss and validation accuracy by epoch"

### *Solution Overview*

Like any other image recognition solution using Convolutional Neural Network (CNN), this solution has the following steps :

1. Load the training, validation and test data
1. Review the data by exploring and visualizing it
1. Augument the data if there are imbalances in the training dataset.
1. Augument the data with fresh images to make the learning more robust
1. Preprocess the data by converting it to gray and normalizing it
1. Define the CNN model architecture.  LeNet has been used as the model for this project
1. Define the training, loss, optimization and preduction functions
1. Train the model using the training dataset and cross validate against the validation dataset.
1. Test the model against the test dataset and new images

### *Data Set Summary & Exploration*

**Dataset Summary** :  Below list provides the summary of the input data

* Number of training images = 34799
* Number of validation images = 4410
* Number of testing images = 12630
* Image data shape = (32, 32, 3)
* Number of unique classes = 43

**Visualization of dataset** :

a. The below bar chart of training data indicates that the images are not evenly spread.  This can skew the learning in favour of images with higher presence.

![alt text][image1]

b. The below bar chart of training and validation labels confirm that both of them have the same number of classes

![alt text][image2]


### *Data Augumentation*

From the training data bar graph, it is evident that the images are not evenly distributed.  The input dataset was enriched with two types of augumentations.

a. **Existing Image Augumentation** : Created new images using images extracted from the training dataset and then jittering them.

b. **New Image Augumentation** : Created 500 new images for each traffic sign using images downloaded from internet and then jittering them.

Below are some of the jittered images.

![alt text][image4]      ![alt text][image5]      ![alt text][image6]      ![alt text][image7]     ![alt text][image8]


Jittering techniques used were Rotation, Translation, Shear and Brightness.  Citation - Reused code for jittering from Vivek Yadav's repo.  Refer 'https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3' for more details.

The below bar graphs shows the image distribution before and after augumentation.  The distribution has been evened out to a reasonable extent.

![alt text][image1]   ![alt text][image3]

---

### *Design and Test a Model Architecture*

#### *Preprocessing*

1. The color image was converted to grayscale to reduce the size of the training dataset, thereby reducing the training time and cost.
1. The grayscaled image was normalized with 0 mean to keep the numbers small


#### *Model Architecture*

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input Layer      		| 32x32x1 Gray image   							| 
| Layer 1 - Conv 5x5  	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Layer 2 - Conv 5x5   	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Layer 3 - Conv 1x1    | 1x1 stride, valid padding, outputs 10x10x128	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128	 				|
| Fully connected 0		| Input = 5x5x128. Output = 3200				|
| Fully connected 1		| Input = 3200 Output = 512						|
| Fully connected 2		| Input = 512 Output = 256						|
| Fully connected 3		| Input = 256 Output = 128						|
| DropOut Layer			| 0.5 Dropout during training					|
| Output layer			| Input = 128 Output = 43						|


#### *Model Training*

Below are the key aspects of training :

* Weights were initialized with a mean of 0 and sigma of 0.1
* Batch size of 128
* Learn rate of 0.001
* 10 Epochs
* AdamOptimizer was used to optimize the model.


#### *Solution Approach*

LeNet architecture has been used as the problem was approached as an image recognition problem.  Below are some of the key considerations :

1. The hyperparameters were tuned first through a trial and error process.  Learning rate of 0.001 and batch size of 128 were found to be optimal for this model.  Epochs and weight considerations are explained below.
1. The model loss was similar for both color and gray images.  As gray images need less training time, therefore being less expensive and quick, decided to gray scale the images.  Normalization was done to lower the range of input numbers that the model had to operate on for calculations.
1. The mean for weight initialization was always kept at zero.  The sigma was found to be sensitive depending on whether the image used was color or gray scale.  0.025 was found to be most effective for training with grayscale images.
1. Experimented with multiple model configurations.  It was observed that increasing the filter depth of the convolutional layer increased the accuracy at the expense of processing time.  The numbers finalized in this model (32, 64, 128) provided good results with reasonable training time.
1. Adding additional convolutional layers increased the accuracy at the cost of training time.  Three layered convolutional network with ReLU has been effective for this project.
1. My intution was that SAME padding with smaller convolutional shapes would provide greater accuracy as details are not lost.  But the results proved otherwise and hence retained VALID padding.
1. Max pooling has been applied to 2 out of 3 convolutional layers to reduce the size of image progressively.  Applying Max pooling to all layers reduced the accuracy without much trade off in training time.  Hence used only 2 Max pooling layers instead of 3.
1. Used 4 fully connected layers (including the output layer).
1. Dropout has been added before the output layer to reduce overfitting.
1. The fine tuned model was run for both 10 and 20 epochs.  The training saturated around epoch 10 and there wasn't significant improvement post that.  Hence used epoch of 10 for final run.
1. The overall training time was ¬24 mins with each epoch taking around 2 min 30 secs.
1. The accuracy was similar across training, validation and test datasets confirming that the model is neither underfitted or overfitted.

My final model results were:
* Training set loss of 0.141567
* Validation set accuracy of 0.978
* Test set accuracy of 0.945

![alt text][image17]

---

### *Performance of model on New Images*

#### *New images and their characteristics*

Below are the five new German traffic signs that were used to test the model.

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]

Each image has varying degree of classification difficulty.
* The 'Speed Limit 30' sign has trees in the background and also includes some parts of the sign post above and below it.
* The 'Right of way' sign is relatively clear and straight forward
* The 'No Entry' sign is skewed and has a blue background.  Should be of moderate complexity.
* The 'Child Crossing' sign is skewed and has trees in the backgroud which makes it a bit tricky.
* The 'Left Ahead' sign is blurred which introduces moderate complexity.

#### *Performance on New Images*

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Child Crossing  		| Child Crossing    							| 
| Left Ahead   			| Left Ahead									|
| Speed Limit 30		| Speed Limit 30								|
| Right of Way     		| Right of Way					 				|
| No Entry  			| No Entry      							|


The model was able to correctly predict 5 of the 5 traffic signs, which gives an accuracy of 100%. The accuracy is seem to be better than then test set for the considered new images.


#### *Model Certainty - Softmax Probabilities*

The top 5 predictions and their corresponding labels are shown below. 

[[  6.43179953e-01   1.90313935e-01   1.53949484e-01   8.84528086e-03   3.11361137e-03]
 [  9.99999762e-01   2.14126416e-07   2.88691571e-10   2.32388234e-10   7.96057248e-11]
 [  9.99193847e-01   7.95350294e-04   4.91315450e-06   4.15528666e-06   1.58116438e-06]
 [  1.00000000e+00   9.28225894e-20   2.15806075e-24   1.80071464e-24   2.36337427e-26]
 [  1.00000000e+00   3.02043967e-18   5.70965295e-20   1.91350851e-20   2.11044632e-21]] 

 [[28 29 30 25 24] 
 [34 38 35 15 22]
 [ 1  2  0  5  4]
 [11 26 30 22 21]
 [17 14 16 34 40]]

**Visual representation of the probabilities**

![alt text][image16]


From the bar graph, it is evident that the model is very sure about all images except the first image.  For the first image, the model is relatively sure that this is a child crossing (probability of 0.6). The top five soft max probabilities for the first image were :

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .643         			| Child Crossing								| 
| .190     				| Bicycle Crossing								|
| .154					| Ice											|
| .008	      			| Road work						 				|
| .003				    | Narrow Right      							|


---

### *Visualizing the Neural Network*

Shown below is the visualization of the first convolutional layer that activated for the 'Left Ahead' sign.  In this layer, the network seems to recogonize the object shape and color contrast (Round sign with blue fill).

![alt text][image14]
![alt text][image15]