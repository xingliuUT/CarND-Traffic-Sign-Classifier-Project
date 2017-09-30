# **Traffic Sign Recognition** 


---


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/hist_full.png "hist"
[image2]: ./images/oneEachType.png "sample"
[image3]: ./images/sample7.png "100imgs"
[image4]: ./images/greyscale.png "gray"
[image5]: ./images/accuracy_lenet_noDropOut.png "nodo"
[image6]: ./images/tuneKeepProb.png "tune1"
[image7]: ./images/tuneBatchSize.png "tune2"
[image8]: ./images/accuracy_lenet_08DropOut.png "model"
[image9]: ./images/newTestImg.png "test"


### Data Set Summary & Exploration

#### Basic summary 

I use `numpy` to get a basic summary of the data:
* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is 32 x 32 x 3.
* The number of unique classes/labels in the data set is 43.

#### Exploratory Visualization

##### Bar Plots 

![alt text][image1]

I plot fractions of images of each Traffic sign type in the training, validation and test sets. The names of sign types are labeled on the x-axis.

It appears that 

1. The data sets don't have the same number of images in each class.
2. But training, validation and test sets have similar proportions of each class.

##### Sample Image

![alt text][image2]

I plot one image from each sign type to get an idea what each sign look like.

##### Images from training set plotted. 

![alt text][image3]

I plot 100 images for the sign : speed limit (100 km/h). 
These images vary in contrast, brightness, as well as the size of the sign.


### Design and Test a Model Architecture

#### Preprocessing

##### Grayscale

As a first step, I decided to convert the images to greyscale because

1. Traffic signs use color, shape and text to convey their meaning. There are no two  sign types that only differ in color. Therefore, convert color into grayscale doesn't lose crucial information.

2. The paper by Sermanet & LeCun reported that their model performs better by using greyscale images instead of color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

##### Normalization

As a second step, I normalized the image data because after greyscaling, each pixel has value between 0 and 255. After normalization, value is between 0 and 1.

##### One-hot encoding

I use one-hot encoding to convert the labels from one vector of values 0 to 42 into a sparse matrix.

#### Final Model Architecture

including model type, layers, layer sizes, connectivity, etc.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   				    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Dropout				| keep_prob = 0.8								|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 			    	|
| Dropout				| keep_prob = 0.8								|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Dropout				| keep_prob = 0.8								|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Dropout				| keep_prob = 0.8								|
| Output				| outputs 43   			    					|
 

#### Training the Model 

I use AdamOptimizer for training. One advantage of AdamOptimizer is that it automatically adapt learning rate, making learning rate not a hyper-parameter that needs to be tuned. I use learning rate of 0.001.

I use batch size of 128 and number of epochs of 100. 

I use 80% keeping rate for all the Dropout layers in my model. It might not be optimal and could be further tuned to increase the accuracy of the model.

#### Result & Approach

Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

##### Result

My final model results were:
* training set accuracy of 0.999.
* validation set accuracy of 0.965. 
* test set accuracy of 0.953.

The final model accuracy v.s. epochs is:

![alt text][image8]

##### Approach


I choose the LeNet-5 architecture. 

I believe it could be applied to the traffic sign classification because

1. It has two convolution layers which could pick up increasingly higher level patterns of the traffic sign images. Convolution layers could work well with traffic signs because both the outer shape (circle or triangle) and the details (simbols, words) of a traffic sign provides important information. ConvNets are good at capturing patterns across a set of training sets with different brightness, contrast, size so the model will learn the features to identify each sign.

2. It succeeded in identifying the digits in the MNIST database. And handwritten digits and traffic sign have similarities in that they are simple symbols with not too complicated shapes.

Besides output dimension adjustments, the main problem with the initial architecture is that it over fits our Traffic sign training data set. Here's a accuracy vs epochs graph with the LeNet-5 architecture. It shows that the training accuracy and validation accuracy both converge after about 5 epochs. The training accuracy is about 99% but the validation accuracy is below 93%. This means that the model overfits the training set.

![alt text][image5]

To improve the model, I add dropout layers after each convolution layer and fully connected layer. The dropout layer is a great way to avoid overfitting because it randomly sets some activations between two layers to be zero and therefore forcing the model to learn redundant representations and avoid overfitting. 

##### Parameter tuning

The keeping probability in the dropout layers is tuned. I use about 10% of the images in the training set and iterate on 50 epochs to test because it's faster than using the full data. I tried for 0.7, 0.75, 0.8, 0.85, 0.9 and found that 0.8 and 0.9 gives me the highest validation set accuracy. I choose keeping probability of 0.8.

![alt text][image6]

The batch size is also tuned using the same procedure. I tried for batch size of 64, 128, and 256 and found that 128 gives me the highest validation set accuracy. I therefore choose batch size of 128 in the full data training.

![alt text][image7]

### Test a Model on New Images

#### New Images

Here are 16 German traffic signs that I found on the web:

![alt text][image9]

The first image might be difficult to classify because ...

#### Type Predictions

Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### Probability Predictions 

Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

