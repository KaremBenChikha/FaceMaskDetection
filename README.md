# FaceMaskDetection

Neural Network Algorithm to detect people wearing face masks and take actions accordingly.

# Links for the project

Check out the full project on Hackster: https://www.hackster.io/karem_benchikha/facemask-detection-88fa1a

See a demo of the projet on YouTube: https://www.youtube.com/watch?v=F2LtUaeCkBM

Download Training Images dataset: https://drive.google.com/file/d/1uKAOauSUqH6wtk_wfpSsjNpju1ralPOI/view?usp=sharing

Visit my personal Website for more Projects:

To build such a project, you have to follow 3 main steps:

# Installation

note : this project works better on Ubuntu 20.04 TLS

Step 0

open terminal, navigate to the project folder, and type :
~~~ 
pip3 install -r requirements.txt
~~~

go and grub a cup of coffee as this will take some time.

Step 1

You will create a neural network model with TensorFlow and will train it on a dataset of both people who are wearing facemasks and people who are not.
The dataset can be downloaded from here: https://drive.google.com/file/d/1uKAOauSUqH6wtk_wfpSsjNpju1ralPOI/view?usp=sharing

Note that the algorithm runs on Jupyter notebooks and requires a lot of GPU power to train the model. However, if you execute my code without changing the Model settings, I can guarantee total confidence of 98%.

Step 2

Here, you will create a face recognition algorithm that will be able to detect facemasks on people's faces using the trained model in the previous step. 
if you don't have the GPU power or the needed dependencies or knowledge to work with neural network models. I have included my pre-trained model that can be used in this step without going by step 1. name of the model fil: mask_detector.model

Step 3

Finally, you will add a simple Serial Command to the facemask detection algorithm   that will order the Arduino to switch LED on or off based on the state of detection.
(you must also deploy the Arduino code on your Ardunio and do the correct wiring)

This project needs the following libraries:

TensorFlow - Keras - imutils - numpy - opencv - matplotlib - scipy - argparse - pyserial