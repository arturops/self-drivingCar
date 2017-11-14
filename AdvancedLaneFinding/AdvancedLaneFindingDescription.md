# Advanced Lane Finding 

The goal of the [Advanced Lane Finding notebook](./Advanced_Lane_Finding.ipynb) is to create a pipeline capable of finding lanes and its curvature, so that a car can drive autonomously on its lane and predict if a curve is approaching.

The steps of this project are:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
2. Apply a distortion correction to raw images
3. Apply a perspective transform to rectify binary image (*"birds-eye view"*) 
4. Use color transforms, gradients, etc., to create a thresholded binary image
5. Detect lane pixels and fit to find the lane boundary
6. Determine the curvature of the lane and vehicle position with respect to center (warp the image back to original to see the lane estimation)
7. Display numerical estimation of lane curvature and vehicle position in the image
8. Create a pipeline and process video

**NOTE:** *After number 3, we repeat  2 and 3 with car images rather than testing with chessboards. This is to allow the proper behavior of our piepeline's perpespective transform for lane projection.* 

[//]: # (Image References)

[Chessboard]: ./camera_cal/calibration2.jpg "ChessboardSample"
[ChessboardUndist]: ./output_images/chessboard_undistortion.png "ChessboardUndistortion"
[ChessboardWarped]: ./output_images/chessboard_undist_warped.png "ChessboardWarped"
[LaneUndist]: ./output_images/car_undistortion.png "LaneUndistortion"
[LaneWarped]: ./output_images/car_undist_warped.png "LaneWarped"
[LaneCurveWarped]: ./output_images/car_curve_warped.png "LaneCurveWarped"
[LaneHLS]: ./output_images/car_hls_thresholded.png "LaneHLS"
[LaneHLSWarped]: ./output_images/car_warped_thresh.png "LaneHLSWarped"
[LaneCurveHLSWarped]: ./output_images/car_curve_warped_thresh.png "LaneCurveHLSWarped"
[SlideWindowPolyfit]: ./output_images/car_curve_polyfit_slide_window.png "SlideWindowPolyfit"
[SlideWindowHist]: ./output_images/car_curve_histogram.png "SlideWindowHistogram"
[LaneFound]: ./output_images/car_curve_lane_found.png "LaneFound"
[LaneData]: ./output_images/car_curve_lane_data.png "LaneData"
[LanePipelineResult]: ./output_images/car_curve_final.png "FinalResult"
[e]: ./output_images/.png "Example"

![alt text][LanePipelineResult]

---
## 1. Camera Calibration

The first step is to calibrate the camera with a set of known patterns. In general, people use chessboard pictures and that is what we used for the camera calibration. 

The camera calibration objectives are:
* Find the distortion coefficients
* Compute the camera intrinsic parameters

In order to calibrate the camera parameters, we need to find the critical points(corners of chessboard squares) in the images, so we use ```cv2.findChessboardCorners()``` and ```cv2.drawChessboardCorners``` to draw the points in an image.

Once we found the critical point we can feed the to ```cv2.calibrateCamera()``` to obtain the intrinsic parameters (*cam_matrix*) and the distortion coefficients (*distortion_coef*).

## 2. Image Undistortion 

We can use the intrinsic parameters and the distortion coeffients to undistort images taken with the camera we calibrated. The function we use is ```cv2.undistort()``` and in this code we have a wrapper for it ```undistortImage()```

![alt text][ChessboardUndist]

Once the chessboard images were successfully undistorted, we can undistort car images using the same distortion coeffients obtained from the calibration.

![alt text][LaneUndist]

It might be hard to notice the image was undistorted. However, if we look at the bottom corners of the images, one can tell the car hood is flatter in the undistorted image.

## 3. Perspective Transform

Once the image distortion is working, all image processing techniques can be applied with a higher confidence in their results.

For lane finding, the curvature is a challenging factor to be detected. That leads to find a way of calculating the curvature from the image. However, the lines of the lanes are flat on the ground and the perspective of the camera on board of a car doesn't help to find curvature. For this scenario, it is very helpful to do a perspective transform and look at the lane lines from the top. This transformation is usually called the "birds-eye view". In order to be able to transform images in this way we need to first obtain the perspective transformation matrix through a selection of points in the image and a desired position of such points after a transformation is applied.

First, we use the chessboard image to do a test and obtain a transformation matrix. Then we will use the same approach for images of the actual road with the car's onboard camera.

The approach consists of:

1. Undistort the image using ```cv2.undistort()```
2. Convert the image to grayscale
3. Find the key points in the image(chessboard corners, lane start and end, etc.)
4. Draw the keypoints
5. Define 4 source points (the outer 4 corners detected in the chessboard pattern, or the 4 points that define right and left lanes)
6. Define 4 destination points (must be listed in the same order as src points. Basically the ROI )
7. Use ```cv2.getPerspectiveTransform()``` to get the **transform matrix M**
8. Use ```cv2.warpPerspective()``` to apply **M** and warp the image to a top-down view

![alt text][ChessboardWarped]

Once the chessboard image was warped successfuly, one can use the same approach for the images from the car's onboard camera and warped the lanes from a ROI. 

Here is an example of straight lanes

![alt text][LaneWarped]

Another example of a curved lane

![alt text][LaneCurveWarped]

The only **additional consideration** is that once the lane is warped, one will process the image and then it must be **unwarped to the original image** to see the lane finding. In order to unwarped an image, one will need to get the **inverse transform matrix Minv** that is also obtained with ```cv2.getPerspectiveTransform()``` by passing backwards the points used to calculate **M** i.e. the 4 source points as destination and the 4 destination points as source. Then, use ```cv2.warpPerspective()``` to apply **Minv** and warp the image back to normal. This caveat will be used after all the image processing in the warped image in step 6.

## 4. Color Thresholding

After the perspective view is giving successful transformations, the next step will be to threshold the image and detect the lanes. The approach in here is using color thresholding by converting the RGB image into HLS and finding the right values to detect the lanes. The function ```colorFilteringHLS()``` does the thresholding.

Example of a whole image from the onboard camera

![alt text][LaneHLS]

Thresholded warped images

![alt text][LaneHLSWarped]

![alt text][LaneCurveHLSWarped]

The lanes are detected by the color thresholding in HLS space. This is very useful as the next steps requires that a thresholded binary image identifies lanes properly.

## 5. Lane Detection - Sliding Window Polyfit

After the color thresholding in HLS space most of the lane is extracted and by using only color thresholding the pipeline will be faster. In case that color is not enough, one can consider magnitude and orientation gradient thresholdings. However, the color served well in here. 

Once we are capable of undistort, change perspective from top-down of lanes and extract the lanes, we need to find the curvature (if any) of the lanes and for that we will use a sliding window method based on pixels histogram of a binary image of the segmented lanes.

The way the histogram sliding window works is:

1. Split in half the warped segmented image of a lane
2. Scan those two halfs with rectangular blocks (windows, shown in green in the image below) 
3. Find the places where there is a higher number of white pixels using a histogram of the image and searching for the max value in a specific neighborhood
4. Collect the right and left halfs max values of each scan
5. Obtain the curve of each lane with ```np.polyfit()``` and left, right points 

The function ```polyfitSlidingWindow()``` taken from [Udacity's Self-Driving Car Program](https://www.udacity.com/)  does the steps above and return the coeffients of the left and right curves in the form Ay^2 + By + C for each curve. 
**IMPORTANT:** We are calculating the curve in function of *y* because the pixels in *x* are very similar for different values of *y*.

The process described above can be seen in the images below, where:
* Left lane equation is *-0.00061y^2 + 0.83891y -61.49462* 
* Right lane equation is *-0.00031y^2 + 0.62448y + 917.72324* 

![alt text][SlideWindowPolyfit]
![alt text][SlideWindowHist]

Notice that the histogram distribution represents the pixels in the curve (look at x axis in both images). 

## 6. Calculate Curvature and Position Respect to the Lane

After computing the equations of each curve, one can calculate the curvature **R** of each lane. The equation one can use to compute such curvature is:

```
    (1 + [f'(y)]^2 )^3/2
R = ---------------------
           f''(y)
```

where

```
f(y)   = Ay^2 + By + C
f'(y)  = 2Ay + B
f''(y) = 2A
```

then

```
    (1 + (2Ay + B)^2 )^3/2  
R = ----------------------
              2A
```

For additional information visit the [curvature tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php).

After the curvature calculation for each lane, we obtained ```R_right``` and ```R_left```. The last step to calculate the lane curvature is to average both curvatures:

```
          R_left + R_right
R_lane = ------------------
                 2
```

`R_lane` however will be in pixels if we do not apply a conversion of pixels to meters or feet. In this case, we follow Udacity's consideration:

*"The lane is about 30 meters long and 3.7 meters wide"* 

With such consideration in mind we can apply conversion factors Metters/Pixels:

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

Assuming the camera is in the middle of the picture and that the lane extends to the length of all x-axis, it is possible to compute the position of the car in the lane. The position of the lane's center using *y_pixels* is the max value of *y*.  We can use that value to calculate the lane in `R_right` and `R_left`, and average both to get the position on the lane center, which is our reference point to compare against the ideal position of the car. The ideal position of the car is the *max value of x* over 2, as we assumed the lane extends through the whole x-axis and the car si the middle.
Finally, the `(ideal_car_position - lane_center) * xm_per_pix` gives the position of the car respect to the center of the lane.

The curvature and the car's position on the lane is computed in `calc_curv_rad_and_center_dist()`. Moreover, it is in this step that we warped the perspective image back to the original image using **Minv**. This is done in `draw_lane()`

![alt text][LaneFound]

## 7. Display the Data

The last step is to show the computed information in the image processed. This is done in `display_data()`.

![alt text][LaneData]

## 8. Pipeline and Results

The pipeline is a series of steps that were described above. The pipeline used was:

| **Advanced Lane Detection Pipeline**        |
|---------------------------------------------|
| Undistortion                                |
| Perspective Transform                       |
| Color Thresholding                          |
| Fitting the Lane's Curve                    |
| Compute Lane's Curvature                    |
| Estimate Car's Position                     |
| Display Lane's Curvature and Car's Position |

After an image from the car's onboard camera passes through the pipeline, one can tell the curvature of the lane and teh car's position in the lane.

![alt text][LanePipelineResult]

Implementing the pipeline in a **video** shows the result from the video link

[![alt text][LaneData]](https://youtu.be/uGAv4svKBKc)

**Click the image above to watch the lane finding drive**

## Lessons Learned and Future Work

During this project, I learned different computer vision techniques such as undistortion, perpective transform, gradient thresholding, color thresholding and histogram sliding window. Other skills learned were fitting curves using python and camera calibration.

The test video was computed sucessfully, however challenge videos are something I left for future work. In order to achieve a better pipeline there are some recommendations to follow:


* Implement a Magnitude and Direction Gradient thresholding for situations where light is very strong and color thresholding fails
* Predict the lane curves without computing the curves in every frame. It can be done by using histogram distribution 
* Calculate the width of the lane to get a more accurate car position in the lane
* If one is more adventorous, I would even attempt to extract unique features that correspond to the lanes and use them in a neural network or SVM to extract the lanes and predict the curvature with machine learning.

