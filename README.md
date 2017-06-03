This is my submission for the Udacity Self-Driving Car Nanodegree Behavioral Cloning Project. You can find my Python code [here](https://github.com/pszczesnowicz/SDCND-P3-BehavioralCloning/blob/master/model.py). The goal of this project was to train a convolutional neural network to drive around a track using data collected from a simulator.

## Data Augmentation

My final model performed the best without any data augmentation. See the Solution Design section for my reasoning.

## Data Pre-Processing

To improve the optimizer efficiency I normalized the input images by subtracting 128 from the pixel values and then dividing the result by 128. By doing so the pixel values would be in the range -1 to 1 with a mean of 0.

`model.add(Lambda(lambda x: (x - 128.0)/128.0, input_shape = image_shape))`

I cropped the input images to remove the car hood and sky which do not add value when predicting the steering angle.

`model.add(Cropping2D(cropping = ((68, 24), (2, 2))))`

These pre-processing operations are inline with the model and are run for both training and driving.

## Model Architecture

I chose to use the NVIDIA End-to-End network architecture because it was developed for almost an identical problem: train a model using three camera inputs and autonomously drive a car using only a single camera input.

My model consisted of the following layers:

| Layer           | Description                                               |
|:----------------|:----------------------------------------------------------|
| Input           | 68x316x3 RGB image                                        |
| Convolution     | 5x5 filter, 2x2 stride, valid padding, output = 32x156x24 |
| ReLU            | Rectified linear unit activation                          |
| Convolution     | 5x5 filter, 2x2 stride, valid padding, output = 14x76x36  |
| ReLU            | Rectified linear unit activation                          |
| Convolution     | 5x5 filter, 2x2 stride, valid padding, output = 5x36x48   |
| ReLU            | Rectified linear unit activation                          |
| Convolution     | 3x3 filter, 1x1 stride, valid padding, output = 3x34x64   |
| ReLU            | Rectified linear unit activation                          |
| Convolution     | 3x3 filter, 1x1 stride, valid padding, output = 1x32x64   |
| ReLU            | Rectified linear unit activation                          |
| Flattening      | Output = 2048                                             |
| Fully connected | Output = 100                                              |
| ReLU            | Rectified linear unit activation                          |
| Fully connected | Output = 50                                               |
| ReLU            | Rectified linear unit activation                          |
| Fully connected | Output = 10                                               |
| ReLU            | Rectified linear unit activation                          |
| Output          | Output = 1                                                |

My final model performed the best without any regularization layers. See the Solution Design section for my reasoning.

## Training Strategy

My final model was trained on data that I collected from both tracks. The lap time ratio for tracks 1 and 2 ended up being roughly 2:1. I wanted both tracks to be equally represented, so I collected 2 laps of data from track 1 and 1 lap from track 2. I also drove the same amount on both tracks but in reverse to collect data that would help the model generalize, i.e. learn on an equal amount of left and right turn data. My final dataset contains 31,446 examples. The following is the distribution of examples per track and a sample of the three camera images that the simulator outputs.

| Track | Direction | Laps | Examples |
|:------|:----------|:-----|:---------|
| 1     | Forward   | 2    | 7755     |
| 1     | Reverse   | 2    | 7575     |
| 2     | Forward   | 1    | 8943     |
| 2     | Reverse   | 1    | 7173     |

<img src="https://raw.githubusercontent.com/pszczesnowicz/SDCND-P3-BehavioralCloning/master/readme_images/left.jpg" width="200"><img src="https://raw.githubusercontent.com/pszczesnowicz/SDCND-P3-BehavioralCloning/master/readme_images/center.jpg" width="200"><img src="https://raw.githubusercontent.com/pszczesnowicz/SDCND-P3-BehavioralCloning/master/readme_images/right.jpg" width="200">

I used all three camera outputs to train the model. To train the model to recover to the center of the lane I used the left and right camera images with a compensation either added or subtracted from the steering angle associated with each image. The simulator's sign convention is negative for left turns and positive for right turns. In order to center the car using the left camera image a compensation had to be added; the opposite is true for right camera images. This works by training the model to steer towards the center of the lane when it only sees one edge of the road.

To improve turning I used three different compensation values in increasing magnitude. The first was applied to images associated with a zero steering angle to recover the car from drifting to the sides of the road. The second was applied to images that were on the inside of a turn, e.g. left camera image while turning left, to prevent the car from oversteering. The third was applied to images that were on the outside of a turn, e.g. right camera image while turning left, to prevent the car from understeering. This taught the model to make sharper turns which was key to successfully completing track 2.

I used Numpy's histogram function to determine the frequency of left turn, straight ahead, and right turn examples in the dataset.

Dataset distribution:
* Number of left turn examples =  11502
* Number of straight ahead examples =  10503
* Number of right turn examples =  9441

At this point the model was still having a hard time with sharp turns, so I cut 50% of the straight ahead examples which increased turning accuracy.

Dataset distribution after subsampling straight ahead examples:
* Number of left turn examples =  11502
* Number of straight ahead examples =  5253
* Number of right turn examples =  9441

I was able to train a successful model using only 10% of my original dataset. Using a smaller dataset allowed me to train much faster and experiment more with different concepts and parameter tunes. I used an 80/20 training/validation example split.

Number of training and validation examples:
* Number of training examples =  2094
* Number of validation examples =  525

The model is trained using an Adaptive Moment Estimation (Adam) optimizer which is a type of gradient descent optimization algorithm.

My model's hyperparameters are:

* Number of epochs = 20
* Batch size = 32
* Learning rate = 0.001

I also implemented a Keras callback which saves only the best epoch as determined by validation loss.

## Solution Design

At the beginning of this project I started with the NVIDIA End-to-End network and began to add features such as data augmentation in the form of image translation to simulate left and right camera images, image flipping to have an equal amount of left and right turn examples, and regularization in the form of dropout layers to prevent the model from overfitting. But with many hours of experimentation I found that the simple model that I started with performed the best in terms of accurate driving and training time.

The left and right camera images cover any benefit that image translation would have added. By driving in both directions I was essentially creating data equal to flipping the images from driving in only one direction with one important difference: Track 2 has a center lane divider. Had I chosen to use image flipping the resulting data would show the car being driven in both the left and right lanes. Training the model with data from both tracks prevented it from overfitting to features only present in one track, e.g. the red and white turn borders in track 1 or the center lane divider in track 2.

## Conclusion

This was a very frustrating project. I spent hours collecting data that I ended up discarding and even more hours training models that would drive into cliffs. But my efforts paid off when my final model completed a full lap on both tracks. I still have a list of ideas that I would like to test before putting this project to rest. These include training on higher resolution images, data augmentation in the form of shadows and random noise, and improved regularization. My goal is to train a model using data from track 1 and have it successfully complete a lap on track 2.

## References

[Udacity Self-Driving Car ND](http://www.udacity.com/drive)

[Udacity Self-Driving Car ND - Term 1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

[Udacity Self-Driving Car ND - Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3)

[NVIDIA - End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
