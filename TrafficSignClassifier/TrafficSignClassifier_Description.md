# **Traffic Sign Recognition using Deep Learning** 

---

[//]: # (Image References)

[SampleTrafficSign]: ./Description_Images/TrafficSign.png "SampleTrafficSign"
[DataHisto1]: ./Description_Images/histogramOriginalData.png "DataHistogram1"
[DataCollage]: ./Description_Images/DataCollage.png "DataCollage"
[RandomImages]: ./Description_Images/OriginalRandomImages.png "RandomImages"
[AugmentedImages]: ./Description_Images/AugmentedRandomImages.png "AugmentedImages"
[DataHisto2]: ./Description_Images/histogramAugmentedData.png "DataHistogram2"
[LeNet]: ./Description_Images/lenet.png "LeNet"
[NewCNN]: ./Description_Images/TrafficSignClassifierCNN.png "NewCNN"
[RealWorldTrafficSigns]: ./Description_Images/ImagesWeb1.png "RealWorldTrafficSigns"
[IdealTrafficSigns]: ./Description_Images/ImagesWeb2.png "IdealTrafficSigns"
[Image1]: ./Test_Images/test_6.png "50km/h_1"
[Image2]: ./Test_Images/test_7.png "50km/h_2"
[Image3]: ./Test_Images/test_18.png "Stop1"
[Image4]: ./Test_Images/test_1.png "CautionAndFace"
[Image5]: ./Test_Images/test_27.png "Stop2"
[Image6]: ./Test_Images/test_25.png "30km/h_1"

[Image7]: ./Test_Images/test_2.png "Caution"
[Image8]: ./Test_Images/test_3.png "30km/h_2"
[Image9]: ./Test_Images/test_4.png "50km/h_3"
[Image10]: ./Test_Images/test_5.png "50km/h_4"
[Image11]: ./Test_Images/test_8.png "60km/h_1"
[Image12]: ./Test_Images/test_9.png "PriorityRoad1"
[Image13]: ./Test_Images/test_16.png "Right-of-way"
[Image14]: ./Test_Images/test_17.png "Roundabout"
[Image15]: ./Test_Images/test_26.png "DoubleCurve"

[Softmax1]: ./Description_Images/Softmax1.png "Softmax1"
[Softmax2]: ./Description_Images/Softmax2.png "Softmax2"
[Softmax3]: ./Description_Images/Softmax3.png "Softmax3"
[Softmax4]: ./Description_Images/Softmax4.png "Softmax4"
[Softmax5]: ./Description_Images/Softmax5.png "Softmax5"
[Softmax6]: ./Description_Images/Softmax6.png "Softmax6"

[e]: ./Description_Images/ "Example"

The project's goal is to create a CNN model capable of classifying traffic signs with an accuracy of 94% in the validation set. You can find a notebook with the code [here](./Traffic_Sign_Classifier.ipynb).

This project contains several auxiliar functions to manipulate, analyze, display and augment data. All of these are useful to understand what you initially have, create new data for your model (if needed) and visualize that new data or analyze your whole data.

In addition, there is a base class for a Convolutional Neural Network (CNN) that can be used as a guideline when designing new CNN's. Also, there are wrapper functions for a more simplistic design of a new model.

![alt text][SampleTrafficSign]


## Approach to Build a Traffic Sign Recognition Neural Network

The steps of this project are the following:

 1. Load the data set 
 2. Explore, summarize, visualize and prepare the data set
 3. Design, train and test a model architecture
 4. Use the model to make predictions on new images
 5. Analyze the softmax probabilities of the new images

All of the steps above can be found in a [Jupyter Notebook](./Traffic_Sign_Classifier.ipynb) that shows in detail their implementation.

 **NOTE:** Some of the implementations here were done from recommendations of [Udacity](https://review.udacity.com/#!/rubrics/481/view).  

---
## 1. Load Data Set

The data used is the [German Traffic Sign data set](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and it has been widely used to train networks in traffic signs recognition.   

---
## 2. Data Set Exploration, Summary, Visualization and Conclusions

### 2.1. Data Exploration

Using the pandas library for a quick overview of the traffic signs data set, one can realize insightful information of the data.

|   Data set type   |   Sample Images   |
|:-----------------:|:-----------------:|
|       Train       |       34799       |
|    Validation     |       4410        |
|       Test        |       12630       |


* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

For a summary of the 43 classes in the dataset, one can look at the [Sign Names file](./LabelDecoder/signnames.csv). In case, one needs a quick overview here is a table of the classes ID and the description of each ID:

|Class ID|               Sign Description                     |
|:------:|:--------------------------------------------------:|
|   0    | Speed limit (20km/h)                               |
|   1    | Speed limit (30km/h)                               |
|   2    | Speed limit (50km/h)                               |
|   3    | Speed limit (60km/h)                               |
|   4    | Speed limit (70km/h)                               |
|   5    | Speed limit (80km/h)                               |
|   6    | End of speed limit (80km/h)                        |
|   7    | Speed limit (100km/h)                              |
|   8    | Speed limit (120km/h)                              |
|   9    | No passing                                         |
|   10   | No passing for vehicles over 3.5 metric tons       |
|   11   | Right-of-way at the next intersection              |
|   12   | Priority road                                      |
|   13   | Yield                                              |
|   14   | Stop                                               |
|   15   | No vehicles                                        |
|   16   | Vehicles over 3.5 metric tons prohibited           |
|   17   | No entry                                           |
|   18   | General caution                                    |
|   19   | Dangerous curve to the left                        |
|   20   | Dangerous curve to the right                       |
|   21   | Double curve                                       |
|   22   | Bumpy road                                         |
|   23   | Slippery road                                      |
|   24   | Road narrows on the right                          |
|   25   | Road work                                          |
|   26   | Traffic signals                                    |
|   27   | Pedestrians                                        |
|   28   | Children crossing                                  |
|   29   | Bicycles crossing                                  |
|   30   | Beware of ice/snow                                 |
|   31   | Wild animals crossing                              |
|   32   | End of all speed and passing limits                |
|   33   | Turn right ahead                                   |
|   34   | Turn left ahead                                    |
|   35   | Ahead only                                         |
|   36   | Go straight or right                               |
|   37   | Go straight or left                                |
|   38   | Keep right                                         |
|   39   | Keep left                                          |
|   40   | Roundabout mandatory                               |
|   41   | End of no passing                                  |
|   42   | End of no passing by vehicles over 3.5 metric tons |

### 2.2. Data Summary

After more data exploration using pandas to calculate summary statistics, one can notice:

* The mean of the data set ID is between 15 and 16
* The standard deviation is 12
* The 50% percentile is 12 

Thus, one can make more conclusions:

* Clearly one can tell that most of the dataset(around 50%) is devoted to images with labels 0 to 12
* The mean of the labels is 15.7. This implies most of the data's labels are smaller than or equal to 15 as the labels range (min = 0 and max = 42) is almost 3 times the mean (i.e. 15)

### 2.3. Data Visualization

As a confirmation of the exploration and summary above it is good practice to plot the data and the desired method was a bar chart that displays the frequency fo each sample image in the dataset.

![alt text][DataHisto1]

After understanding the number of the data, one would like to see the actual data to understand how it looks like to prepare it for the network to train or even to design a model proper for such data. The following image shows some random images form the data set.

![alt text][DataCollage]

### 2.4. Data Conclusions

From the statistics above and the visualization one can right away believe the data set is very small for a neural network to fully train on it and generalize well. Data Augmentation is a technique that allows to create new images based on the existing ones. It consists on rotating, translating, cropping, flipping, chaging contrast/brightness, etc on an existing image, so that it seems it is seeing from different perspective/conditions that can be beneficial for the CNN to learn. It is also very important to perform such changes to the image using an Affine transformation so the original image doesn't get (statistically) distorted.
In this project, rotation, translations, zooming and shearing techniques were used.

An example of such data augmentation is shown below
##### Original Images
![alt text][RandomImages]

##### Augmented Images
![alt text][AugmentedImages]

After applying augmentation the dataset increase 3 times is size for training and allows the CNN to generalize better. The training data set size was 139196 sample images.

One can see that in the histogram below

![alt text][DataHisto2]

---
## 3. Design and Test a Model Architecture

#### 3.1. Data Preparation 

After analyzing the data set, we concluded that it was rather small to properly train a CNN. Thus, the whole train data set was augmented using data augmentation techiques as an early first step in data preparation.

As a last preparation data step, I decided to normalize the RGB images using the equation:

```python
mean_pixels = 128
standard_dev_pixels = 128
normalized_image = (image - mean_pixels)/standard_dev_pixels
```

The reason behind normalization is that it helps the model converge better to the global minima and as such it will generalize better.

**NOTE:**I decided not to use grayscale as I couldn't see any improvement when used.

#### 3.2. Model Design

Before starting the module, I decided to create wrapper functions:

`fully_connected()` that wraps the fully connected layer and allows to use batch normalization on it, as well as any activation function including a simple linear function.

`convolution_layer()` that wraps convolutional layers and add batch normalization to them. It as well allows any activation function passed to the function.  

The model design in general was one of the most challenging and at the same time the one I decided to make as modular as possible. Thus, I created a base class that gives a guideline to create new modules and has an embedded ```train()``` and ```evaluate()``` functions, so that people who creates a child model from this ```class BaseCNN``` do not need to recreate the logic of how to train a model.

As an example of the model easiness to be used, I implemented LeNet first as shown in the image below

![alt text][LeNet]

As for the model used in this, I decided to test several options:

* Use Batch Normalization to accelerate the training, allows higher learning rates and improves the results of the CNN. Details on [Batch Normalization](https://arxiv.org/abs/1502.03167)
* Use Drop out from ranges to 0.3 to 0.8, to avoid overfitting
* Different number of convolutional layers
* Different padding types: SAME and VALID
* Different Max Pool sizes
* Different sizes for the units in the fully connected layers

After several test, I came up with a few different models, but I decided to use one that quickly achieved a 90% range validation accuracy.

My final model's architecture consisted of the following layers:

![alt text][NewCNN]

A more detailed description is in the table below

| Layer         	|               Description	        	             | 
|:-----------------:|:--------------------------------------------------:| 
| Input         	| 32x32x3 RGB image   							     | 
| Convolution       | 9x9 kernel, 1x1 stride, SAME pad, output 32x32x3 	 |
| Batch Norm        |                                                    |
| RELU				|												     |
| Convolution       | 3x3 kernel, 1x1 stride, SAME pad, output 32x32x16  |
| Batch Norm        |                                                    |
| RELU              |                                                    |
| Max pooling	    | 3x3 kernel, 1x1 stride, SAME pad, output 32x32x16  |
| Convolution       | 5x5 kernel, 3x3 stride, VALID pad, output 10x10x64 |
| Batch Norm        |                                                    |
| RELU              |                                                    |
| Max pooling       | 3x3 kernel, 1x1 stride, VALID pad, output 8x8x64   |
| Convolution       | 3x3 kernel, 1x1 stride, SAME pad, output 8x8x128   |
| Batch Norm        |                                                    |
| RELU              |                                                    |
| Convolution       | 3x3 kernel, 1x1 stride, SAME pad, output 8x8x64    |
| Batch Norm        |                                                    |
| RELU              |                                                    |
| Max pooling       | 3x3 kernel, 1x1 stride, VALID pad, output 6x6x64   |
| Flat              |                                                    |
| Fully connected   | 1224 neurons                                       |
| Dropout           |                                                    |
| Fully connected   | 1224 neurons                                       |
| Dropout           |                                                    |
| Fully connected   | 43 neurons                                         |
| Softmax           |                                                    |


#### 3.3. Model Training Parameters Description 

Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:

* Adam Optimzer, as it has an embbeded momentum that gets faster to minima
* Learning rate of 0.001
* Batch size of 128 images
* 80 epochs
* Dropout of 0.5


#### 3.4. Model Training Procedure

Recalling that the goal of the project is to achieve 0.93 at least in the validation accuracy. I decided to start with what I was familiar and then move to learn new CNN's.

I looked at some well known achitectures such as LeNet, AlexNet and VGG to try to come up with my own architecture. I found that is was harder than I expected it to be, but it helped me learn a lot during the process. Once resource I recommend to everyone to read is [CNN papers you must know](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html). This resource helped me understand better CNN skimming through some of the papers recommended here. Then I decided to use AlexNet as a start point since it won the Imaginet challenge in 2012, it seemed to me to be a very simple model and quite comparable to LeNet with which I was familiar already.

I did not get good results at first and it was an iterative approach based on educated changes to the model. At first, I used LeNet and only used batch normalization in all layers. I had read about batch nromalization and it minimized the use of dropout and it was not majorly impacted by the initialization of weights as long as they were in a reasonable range the model will correct itself. Also, it helps to increase the training speed, so I was hoping that I could get 94% validation accuracy just like that, but that was not the case. Then, I started to experiment with dropout as I had the feeling my network was overfitting (a high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting) since my validation and test accuracy was very low for what I had in the training accuracy, but I was not getting any good result and it seemed that the performance of the layer actually went down. That is when I decided I needed to look for some other options models and not just stick to LeNet. After skimming through some papers I decided to quickly implement AlexNet and I could see it was performing a bit better tha LeNet, but it wa snot lose at all to good. I started tunning parameters and running the model:

 * Changing hyperparameters
 * Increasing/Decreasing the number of filters in the convolutional layer
 * Adding/Reducing features in the fully connected layers
 * Add/Remove layers

I iterated to that process until I finally noticed the model was performing a lot better than when it started. It was finally reacing more than 90% accuracy in the validation set very easily and fast.

At some point I realized I could keep experimenting forever, so I decide to stick to the model I described above.

My final model results were:

* Training set accuracy of 0.999
* Validation set accuracy of 0.955 
* Test set accuracy of 0.954

---
## 4. Test a Model on New Images

#### 4.1. Web Images

I chose 72 German traffic signs found on the web and used them to test the generalization of the model. I had to preprocess the images using Mac OS X Preview. I cropped and resized images to be 32x32, so that the model could process them. Out of those 72 images:

* 29 Images are cropped from pictures of real world scenarios with different angles and illumination

![alt text][RealWorldTrafficSigns]

* 43 Images cropped from a picture of how all traffic signs look like on paper

![alt text][IdealTrafficSigns]

The 29 images set from real scenarios will be harder to detect as conditions can be unseen or pictures can have more elements into them that cause the CNN to fail recognizing the traffic sign. Here are five of the images that because of distance, additional elements or illumination can make the CNN to fail in its prediction:

![alt text][Image1] ![alt text][Image2] ![alt text][Image3] ![alt text][Image4] ![alt text][Image5] ![alt text][Image6]

However, there will also be images that have a hard degree of detection, but the CNN can get a good chance to predict correctly thanks to the augmented data. Some of those images are:

![alt text][Image7] ![alt text][Image8] ![alt text][Image9] ![alt text][Image10] ![alt text][Image11] ![alt text][Image12] ![alt text][Image13] ![alt text][Image14] ![alt text][Image15]    


#### 4.2. Web Images Accuracy

**The 29 real world conditions web images**
The model was able to correctly guess 20 of the 29 traffic signs, which gives an accuracy of 71%. This result is very reasonable as the images it couldn't guess have very different points of view or more artifacts than the ones in the training,validation and test datasets. 

**The other 43 web images**
The model was able to correctly guess 36 of the 40 traffic signs, which gives an accuracy of 95%. This result is amazing as this images are similar but not the same as the ones in the training,validation and test datasets. 

These results verify that the network model is training well and that more data from different points of view or a more complex data augmentation can definitively improve how the model generalizes.


#### 4.3. Softmax Probabilities of Web Images

In order to display some of the performance in the images chosen from the web that are more similar to the real world, I added some of the softmax examples from the notebook in here

![alt text][Softmax1] 

![alt text][Softmax2]

![alt text][Softmax3] 

![alt text][Softmax4]

![alt text][Softmax5] 

![alt text][Softmax6]

After looking at the examples it is very clear that for images that are similar to the train data the network had almost no issue predicting it correct most of the time, while for images that are very far or where illumination is partial the model is not quite accurate as it has not been trained on images of such type or with such artifacts. 

### Future Work - Visualizing the Neural Network (See the Ipython notebook for more details)
Discuss the visual output of the trained network's feature maps. What characteristics did the neural network use to make classifications?


