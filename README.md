**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_from_web/german_traffic_sign_1.png "Traffic Sign 1"
[image2]: ./images_from_web/german_traffic_sign_2.png "Traffic Sign 2"
[image3]: ./images_from_web/german_traffic_sign_3.png "Traffic Sign 3"
[image4]: ./images_from_web/german_traffic_sign_4.png "Traffic Sign 4"
[image5]: ./images_from_web/german_traffic_sign_5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dhkim0225/CarND-Traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and basic python operations to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Preprocessing pipeline has 4 steps.

1. Make image grayscale.
	It's for easily learning data.

2. Normalization with MinMaxScaler(scikit-learn).
	It's for efficiency of optimizer.

3. For making ndarray to tensorflow, I use tolist() function to change data types.

4. Shuffle data.
	It's for efficiency of optimizer.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x3 RGB image 				| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 16x16x8 			|
| Convolution 5x5	| 1x1 stride, same padding, outputs 16x16x16 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 8x8x16 			|
									|
| Fully connected	| W.shap (8*8*16, 300), dropout_rate(learn) 0.7 |
| Fully connected	| W.shap (300, 120), dropout_rate(learn) 0.7	|
| Fully connected	| W.shap (120, 43), dropout_rate(learn) 0.7 	|

| Softmax		|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

It's trained with Adamoptimizer. And by splitting the data by batch_size, It was able to increase the efficiency of Adam optimizer even further.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* cost = 0.012
* validation set accuracy of 0.953 
* test set accuracy of 0.936

At first, I simply designed the conv layer to be deeper and wider for good accuracy.
But unlike expectations, I couldn't get much better results than I thought.

So I designed the conv layer a little shallower and narrower, applied max pooling and applied dropout to the fully connected layer. I wanted to apply it to an ensemble, but I did not apply it because I had a lot of other things to do.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The third image might be difficult to classify because my network confused it as 80.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        	| 
|:-----------------------------:|:-------------------------------------:| 
| No entry      		| No entry 				| 
| Stop    			| Stop					|
| Speed limit (60km/h)	   	| Speed limit (80km/h)			|
| Go straight or left	      	| Go straight or left			|
| Go straight or right		| Go straight or right 			|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 1         		| No entry  					| 
| 4.7e-15     		| stop	 					|
| 2.2e-17		| Speed limit (20km/h)				|
| 2.2e-19     		| Speed limit (120km/h)				|
| 5.8e-22		| Go straight or right     			|

For the second image

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         		| Stop  					| 
| 2.3e-5     		| Speed limit (70km/h)	 					|
| 1.2e-5		| Speed limit (80km/h)				|
| 9.9e-7     		| Turn right ahead				|
| 6.7e-7		| Speed limit (20km/h)     			|

For the third image

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.97         		| End of speed limit (80km/h)  			| 
| 0.02     		| Speed limit (30km/h)	 			|
| 5.7e-3		| Speed limit (60km/h)				|
| 2.4e-4     		| Keep right					|
| 1.1e-5		| End of all speed and passing limits 		|

For the fourth image

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.97         		| Go straight or left 				| 
| 2.7e-2     		| Ahead only 					|
| 5.7e-3		| Keep left					|
| 2.4e-4     		| Priority road					|
| 1.1e-5		| Yield     					|

For the fifth image

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         		| Go straight or right  			| 
| 1.8e-2     		| Keep right	 				|
| 2.2e-6		| End of all speed and passing limits		|
| 6.1e-7     		| Priority road					|
| 4.0e-8		| No entry		     			|


