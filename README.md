[image1]: ./images_from_web/german_traffic_sign_1.png "Traffic Sign 1"
[image2]: ./images_from_web/german_traffic_sign_2.png "Traffic Sign 2"
[image3]: ./images_from_web/german_traffic_sign_3.png "Traffic Sign 3"
[image4]: ./images_from_web/german_traffic_sign_4.png "Traffic Sign 4"
[image5]: ./images_from_web/german_traffic_sign_5.png "Traffic Sign 5"

# Traffic Sign Classifier
Udacity Self Driving Car Nanodegree project 2.
This project classify traffic sign with simple Convolutinon Network.

## Dataset
I used the numpy and basic python operations to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43 

## Preprocessing
Preprocessing pipeline has 3 steps.

1. GrayScaling
2. Normalization with MinMaxScaler(scikit-learn)
3. Shuffle data.

## Model Architecture
| Layer            | Description                                   | 
|:----------------:|:---------------------------------------------:| 
| Input            | 32x32x3 RGB image                             | 
| Convolution 5x5  | 1x1 stride, same padding, outputs 32x32x8     |
| RELU		       |                                               |
| Max pooling	   | 2x2 stride,  outputs 16x16x8                  |
| Convolution 5x5  | 1x1 stride, same padding, outputs 16x16x16    |
| RELU		       |                                               |
| Max pooling	   | 2x2 stride,  outputs 8x8x16                   |
| Fully connected  | W.shap (8*8*16, 300), dropout_rate(learn) 0.7 |
| Fully connected  | W.shap (300, 120), dropout_rate(learn) 0.7	   |
| Fully connected  | W.shap (120, 43), dropout_rate(learn) 0.7 	   |
| Softmax	       |                                               |

## Result
* cost = 0.013
* validation set accuracy = 0.949
* test set accuracy = 0.942

## Test
Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

Here are the results of the prediction:

| Image			        |     Prediction	        	| 
|:-----------------------------:|:-------------------------------------:| 
| No entry      		| No entry 				| 
| Stop    			| Stop					|
| Speed limit (60km/h)	   	| Speed limit (60km/h)			|
| Go straight or left	      	| Go straight or left			|
| Go straight or right		| Go straight or right 			|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 1         		| No entry  					| 
| 1.1e-25     		| stop	 					|
| 2.5e-28		| Yield						|
| 5.9e-31     		| End of all speed and passing limits		|
| 2.8e-33		| Speed limit (120km/h)     			|

For the second image

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.84         		| Stop  					| 
| 0.13     		| Yield	 					|
| 7.6e-3		| No vehicles					|
| 4.4e-5     		| Speed limit (50km/h)				|
| 2.1e-6		| Speed limit (60km/h)     			|

For the third image

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         		| Speed limit (60km/h)  			| 
| 1.0e-5     		| Speed limit (80km/h)	 			|
| 2.2e-7		| End of speed limit (80km/h)			|
| 2.0e-8     		| Speed limit (50km/h)				|
| 2.4e-9		| Speed limit (30km/h)		 		|

For the fourth image

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         		| Go straight or left 				| 
| 1.4e-2     		| No entry 					|
| 1.3e-5		| Keep left					|
| 6.8e-8     		| Roundabout mandatory				|
| 6.8e-12		| Ahead only     				|

For the fifth image

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.80         		| Go straight or right  			| 
| 0.15     		| No entry	 				|
| 0.05			| Keep right					|
| 8.5e-4     		| End of all speed and passing limits		|
| 7.6e-4		| End of no passing	     			|


