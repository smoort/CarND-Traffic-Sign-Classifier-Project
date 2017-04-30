# **Traffic Sign Recognition Project Writeup**

### This is a write up on the German Traffic sign recognition project by *Saravanan Moorthyrajan*

---

### **Goals of the Traffic Sign Recognition Project**

* Load the provided data set
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

### *Solution Overview*

Like any other Convolutional Neural Network (CNN) based image recognition, the solution has the following steps :

1. Load the training, validation and test data
1. Review the data by exploring and visualizing it
1. Augument the data if there are imbalances in the training dataset.  This is done to ensure that the learning of the network is not skewed towards classes with large number of training images
1. Augument the data with fresh images to make the learning more robust
1. Preprocess the data by converting it to gray and normalizing it
1. Define the CNN model architecture.  LeNet has been used as the model for this project
1. Define the training, loss, optimization and preduction functions
1. Train the model using the training dataset and cross validate against the validation dataset.
1. Test the model against the test dataset

### *Data Set Summary & Exploration*

**Dataset Summary** :  Below list provides the summary of the input data

* Number of training images = 34799
* Number of validation images = 4410
* Number of testing images = 12630
* Image data shape = [(32, 32, 3)]
* Number of unique classes = 43

**Visualization of dataset** :

a. The below bar chart of training data indicates that the images are not evenly spread.  This can skew the learning in favour of images with higher presence.

![alt text][image1]

b. The below bar chart of training and validation labels confirm that both of them have the same number of classes

![alt text][image2]


### *Data Augumentation*

From the training data bar graph, it is evident that the images are not evenly distributed.  The input dataset was enriched with two types of augumentations.

**Existing Image Augumentation** : Created new images using images extracted from the training dataset and then jittering them.

**New Image Augumentation** : Created new images using images downloaded from internet and then jittering them.

Below are some of the jittered images.

![alt text][image4]      ![alt text][image5]      ![alt text][image6]      ![alt text][image7]     ![alt text][image8]


Jittering techniques used were Rotation, Translation, Shear and Brightness.  Citation - Reused code for jittering from Vivek Yadav's repo.  Refer 'https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3' for more details.

The below bar graphs shows the image distribution before and after augumentation.  The distribution has been evened out to a reasonable extent.

![alt text][image1]   ![alt text][image3]


### *Design and Test a Model Architecture*

### *Preprocessing*

1. The color image was converted to grayscale to reduce the size of the training dataset, thereby reducing the training time and cost.
1. The grayscaled image was normalized to have 0 mean to keep the numbers small


### *Model Architecture*

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Layer 1 - Conv 5x5  	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Layer 2 - Conv 5x5   	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Layer 3 - Conv 1x1    | 1x1 stride, valid padding, outputs 10x10x128	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128	 				|
| Fully connected		| Input = 5x5x128. Output = 3200				|
| Fully connected		| Input = 3200 Output = 512						|
| Fully connected		| Input = 512 Output = 256						|
| Fully connected		| Input = 256 Output = 128						|
| Fully connected		| Input = 128 Output = 43						|
|						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


