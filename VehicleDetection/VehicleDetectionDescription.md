# Vehicle Detection

The goal of this project is to generate a succesful pipeline that allows to detect cars from image and video taken from a camera placed on the front center of a car. The code for this project is implemented in the [Vehicle_Detector notebook](./Vehicle_Detector.ipynb).

The steps to achieve the goal are:

1. Find and analyze a dataset of cars and non-cars images to train a Linear SVM with a test accuracy greater than 96% when finding car from images 
2. Process images and extract features from them. In this case, the chosen feature extraction methods are:
    * Color Histogram features extraction
    * Spatial features extraction
    * Histogram of Orientation Gradient (HOG)
3. Train the linear SVM with the extracted features and verify its test accuracy is greater than 96%
4. Implement a sliding window in images to look only in region of interst (ROI) for potential features that give a positive car prediction
5. Combine multiple detection and eliminate false positives by using a heat-map 
6. Create a pipeline with all the elements above and process video

At the very end a small discussion on improvements will describe in high level how the approach taken here can be improved

[//]: # "Image References"
[Dataset]: ./output_images/data_visualization.png "Dataset"
[ColorHistogram]: ./output_images/color_hist_vis.png "ColorHistogram"
[SpatialBinImgs]: ./output_images/bin_spatial_images.png "SpatialBinningImages"
[SpatialBinFeatsOriginalImg]: ./output_images/bin_spatial_feats_original.png "SpatialFeaturesOriginalImage"
[SpatialBinFeatsResizedImg]: ./output_images/bin_spatial_feats_resized.png "SpatialFeaturesResizedImage"
[HOG]: ./output_images/hog_img.png "HOG"
[SlidingWindow]: ./output_images/sliding_window.png "SlidingWindow"
[CarsWindowsDetection]: ./output_images/cars_windows_detection.png "CarsWindowsDetection"
[HeatmapMultipleSuccess]: ./output_images/heat_map0.png "HeatmapMultipleSuccess"
[HeatmapEmptySuccess]: ./output_images/heat_map1.png "HeatmapEmptySuccess"
[HeatmapThreshFail]: ./output_images/heat_map2.png "HeatmapThresholdFail"
[HeatmapMultipleSuccess2]: ./output_images/heat_map3.png "HeatmapMultipleSuccess2"
[PipelineResult]: ./output_images/pipeline_img.png "PipelineResult"

![alt text][HeatmapMultipleSuccess2]

## 1. Dataset

The dataset used in here was taken from several places and compiled by Udacity into two datasets:

* 8792 [Vehicles_images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
* 8968 [Non-vehicles images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

![alt text][Dataset]

The images visualizations used are inspired and taken from [ILYAmLV repo](https://github.com/ILYAmLV/CarND-Vehicle-Detection-and-Tracking)

## 2. Feature Extraction

The need to identify unique features of the cars is essential to use machine learning to distinguish from cars and non-cars.

### 2.1. Color Features

Color features can be obtained using a color histogram from the whole image. The distribution of colors in the histogram can help us recognize what can be a car as particular colors are used for car and not in the road.

We use ```color_hist()``` from the ```feature_extraction.py``` module to extract the color features using a histogram of the image

![alt text][ColorHistogram]

Color features can be obtained using a color histogram from the whole image. The distribution of colors in the histogram can help us recognize what can be a car as particular colors are used for car and not in the road.

### 2.2. Spatial Binning

The spatial features will help to confirm that color features will not change even if the car image is a bit compressed or the car is far off and a low resolution picture is taken.

We use ```bin_spatial()``` from the ```feature_extraction.py```  to extract the features from two images of size 64x64 and 16x16. 

![alt text][SpatialBinImgs]
![alt text][SpatialBinFeatsOriginalImg]
![alt text][SpatialBinFeatsResizedImg]

One can see that depending on which color space an image is it has different features. However, if the images have different sizes, they tend to have similar features. I decided to use:

```python
* YCrCb color space features because they showed a better accuracy in the SVM training
* spatial_size = (16, 16) # Spatial binning dimensions
* hist_bins = 32          # Number of histogram bins
```

### 2.3. Histogram of Orientation Grandient (HOG)

The color features seem to be a potential method for cars' detection, however, we need a method that is color independent. The method that allows such capability is a histogram of orientation gradient which will mainly give the shape of objects and is extensively used in computer vision. Extraction of HOG features will be key for our SVM to learn how to accurately detect cars.

We use ```get_hog_features()``` from ```feature_extraction.py``` to extract the HOG features.

The features used for HOG extraction are:

```python
* orient = 9          # HOG orientations
* pix_per_cell = 8    # HOG pixels per cell
* cell_per_block = 2  # HOG cells per block
* hog_channel = "ALL" # Can be 0, 1, 2, or "ALL" channels
```

![alt text][HOG]

It is **important** to know that the HOG can only be applied to a single color channel. In this the image above, we used a grayscale image to prove the concept of HOG. However, during the actual HOG extraction we extract feature per channel of the color space.

## 3. SVM Training

Once we extract features independetly, there is the need to extract all at once if desired so that the SVM can properly identify cars. The function ```extract_features()``` in ```feature_extraction.py``` computes all the features of an image. This function is used to train the SVM with the parameters mentioned on the feature extraction section. The SVM is trained by calling ```vechicle_classifier()``` in ```SVMcar_Classifier.py``` and it creates a binary pickled file that stores the configuration used to train the SVM.

I decided to use a Linear SVM because a polynomial SVM would perform slower and my pipeline already is slow. 

The result of the training was a ```Test accuracy of 98.87%``` which is greater than the goal of 96%.

## 4. Sliding Window 

Once the SVM has trained properly in the dataset, we need to implement a technique known as sliding window to be able to scan bigger images and look for little windows in a big image where the image in the window can be classified as a car or non-car.

We basically put a ROI in the image and we create a grid in the ROI. We use ```slide_window()``` and ```search_windows()``` from ```feature_extraction.py``` to show how this works, but in the actual pipeline we use a different piece of code to speed up this technique.

![alt text][SlidingWindow]

After setting up the grid in the ROI we can detect if each window contains a car or non-car.

![alt text][CarsWindowsDetection]

The detections are shown in squares (windows) of different color.

## 5. Heatmap

A single car can be detected by different windows and there can be false positives. Thus, we need to add a heat map technique to make the detection robust and to combine multiple detections into a single detection based on how many times (threshold) a specific area in the picture was detected.

Example of multiple detections combine into one
![alt text][HeatmapMultipleSuccess]

Example of none detection when there is no car
![alt text][HeatmapEmptySuccess]

Example of threshold number of detections not met for an actual car detection
![alt text][HeatmapThreshFail]

## 6. Pipeline

Finally, we put together all the techniques described above to make pipeline

| **Vehicle Detection Pipeline**              |
|---------------------------------------------|
| Convert RGB to YCrCb                        |
| Slide Window in the full image              |
| Extract HOG features                        |
| Extract Color features                      |
| Extract Spatial features                    |
| Predict with SVM                            |
| Heatmap                                     |
| Display car detections                      |

After an image or frame passes through the abve pipeline we get the result below

![alt text][PipelineResult]

## Lessons Learned and Future Work

During this project, I learned a variety of image processing and feature extraction techniques such as color histogram, spatial binning and HOG. The most outstanding lesson learned for me during this lesson is the sliding window and the combination of it with a linear SVM to detect cars or non-cars windows. I definitively learned a lot from it.

Something that happened in this project was that the more robust I tried to do the detections the slower the pipeline became. This is something that can be improved.

Other improvements can be:

* Use more images in the dataset. A larger dataset with more variety in the weather scenes, car types, sections of cars, etc can help to recognize cars better
* Clean and properly divide some images in the dataset. I noticed that some parts of the cars were actually in the non-cars dataset and those can be moved to the cars' images because sometimes a window only catches a cut section of a car and it does not recognizes it as a car due to the training.
* This is a more advanced idea, but in the real world this detection can be combined with LiDAR data to increase the detections confidence and not only depend on camera images as cameras can be tricked by dark, glare, etc.
* If one is adventurous, adding a CNN with data augmentation and batch normalization might be better at recognizing cars than a linear SVM, the problem will be if the predictions can be done as fast for each window.
* Finally, I believe that the sliding window is a quite powerful, but slow method to detect cars in real time, so I will propose to train an additional SVM or CNN with bigger pictures where there are many cars in the lane, or only a car on the left or right lane, etc. The goal of this other machine learning classifier will be to determine where the sliding window should be applied. For example, the SVM can predict there is a car in the right and the we do a fine sliding window and our current SVM into only that section of the road. That should speed things up by looking only when the classifier think it might be a car and where it should look in detail for it.



