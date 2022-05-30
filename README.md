# Face-Emotion-Recognition-Deep-Learning-Project

# 1 Project Introduction

The Indian education landscape has been undergoing rapid changes for the past 10 years owing to
the advancement of web-based learning services, specifically, eLearning platforms.
Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India
is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market
is growing on a rapid scale, there are major challenges associated with digital learning when
compared with brick and mortar classrooms. One of many challenges is how to ensure quality
learning for students. Digital platforms might overpower physical classrooms in terms of content
quality but when it comes to understanding whether students are able to grasp the content in a live
class scenario is yet an open-end challenge.

In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the
class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who
need special attention. 
Digital classrooms are conducted via video telephony software program (exZoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance.

While digital platforms have limitations in terms of physical surveillance but it comes with the power of
data and machines which can work for you. It provides data in the form of video, audio, and texts
which can be analysed using deep learning algorithms. 
Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no
longer in the teacher’s brain rather translated in numbers that can be analysed and tracked.

![face emotion recognition image1](https://user-images.githubusercontent.com/60994606/167482504-24ad27cb-e1d4-423b-87df-ed140691f457.jpeg)


# 2 Problem Statements

We will solve the above-mentioned challenge by applying deep learning algorithms to live video data.
The solution to this problem is by recognizing facial emotions.

-----------------------------------------------------------------------------------

# Dataset:-

There are many datasets to work on the this project but we have chosen this one which can be downloaded from the link below through kaggle-

https://www.kaggle.com/datasets/msambare/fer2013
------------------------------------------------------------------------------
# Different Models Used:-

The different models that i have used are ae follows:-

* Deepface

* CNN (Convolutional neural network)

* Xception

-----------------------------------------------
# DeepFace:-

DeepFace is a deep learning facial recognition system created by a research group at Facebook. It identifies human faces in digital images. The program employs a nine-layer neural network with over 120 million connection weights and was trained on four million images uploaded by Facebook users.The Facebook Research team has stated that the DeepFace method reaches an accuracy of 97.35% ± 0.25% on Labeled Faces in the Wild (LFW) data set where human beings have 97.53%. This means that DeepFace is sometimes more successful than the human being.

------------------------------------------------------------


# Xception

Xception architecture is a linear stack of depth wise separable convolution layers with residual connections. This makes the architecture very easy to define and modify; it takes only 30 to 40 lines of code using a high level library such as Keras or Tensorflow not unlike an architecture such as VGG-16, but rather un- like architectures such as Inception V2 or V3 which are far more complex to define. An open-source implementation of Xception using Keras and Tensorflow is provided as part of the Keras Applications module2, under the MIT license. We used Adam as our optimizer after training for 70 epochs using Adam and a batch size of 785, we achieved 64% accuracy on the test set.

![image](https://user-images.githubusercontent.com/60994606/170902341-a4c1836e-919b-47d4-90c5-356e99a97fc3.png)
----------------------------------------------------------------

# Using Transfer Learning Resnet 50:-

Since the FER2013 dataset is quite small and unbalanced, we found that utilizing transfer learning significantly boosted the accuracy of our model. ResNet50 is the first pre-trained model we explored. ResNet50 is a deep residual network with 50 layers. It is defined in Keras with 175 layers. We replaced the original output layer with one FC layer of size 1000 and a softmax output layer of 7 emotion classes. We used Adam as our optimizer after training for 50 epochs using Adam and a batch size of 785, we achieved 63.11% accuracy on the test set and 67% on the train set. There is much less over-fitting. We have taken epochs as 50. Once the threshold is achieved by the model and we further tried to train our model, then it provided unexpected results and its accuracy also decreased. After that, increasing the epoch would also not help. Hence, epochs play a very important role in deciding the accuracy of the model, and its value can be decided through trial and error.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Custom Deep CNN:-

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

We designed the CNN through which we passed our features to train the model and eventually test it using the test features. To construct CNN we have used a combination of several different functions such as sequential, Conv (2D), Batch Normalization, Maxpooling2D, Relu activation function, Dropout, Dense and finally softmax function.

We used Adam as our optimizer after training for 70 epochs using Adam with minimum learning rate 0.00001 and a batch size of 785, we achieved 69 % accuracy on the test set and 74% as train accuracy.

One drawback of the system is the some Disgust faces are showing Neutral .Because less no. of disgust faces are given to train .This may be the reason.
I thought it was a good score should improve the score.

Thus, I decided that I will deploy the model.

![image](https://user-images.githubusercontent.com/60994606/170906386-32df8688-5d1b-4bb3-ac8a-7775be58f4ab.png)


-----------------------------------------------------------------------------------------
