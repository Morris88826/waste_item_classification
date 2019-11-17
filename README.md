# Waste Item Classification
**Mu-Ruei Tseng**, **Tara Poteat**, **Ankit Verma**




## FINAL REPORT****






## Introduction
Managing waste efficiently is a significant challenge. In 2015, 262 million tons of solid waste were processed in the United States alone. 91 million tons of this waste was recycled[1]. Clearly, recycling plays a large role in waste management. The most prominent recycling system in use currently is the Single Stream Recycling System, which involves manual item sorting by humans[2]. However, classifying waste manually is both time inefficient and space inefficient. This leads us to believe that machines will soon have a greater responsibility in classifying waste items. Through this project, we will develop a model that leverages machine learning algorithms to identify an item’s ‘waste-type’ based on an input image.

![](https://i.imgur.com/BVPqZ6f.png)


![](https://i.imgur.com/cqw98ou.png)


![](https://i.imgur.com/x3BzjfA.png)

source:http://www.workforcezone.net/turnover/

source: http://datatopics.worldbank.org/what-a-waste/trends_in_solid_waste_management.html

## Dataset

dataset source: https://www.kaggle.com/asdasdasasdas/garbage-classification

total of 2527 images of waste items that each fall into one of the following class: cardboard, glass, metal, paper, plastic, or trash. 

The dataset's count of images by class label is shown below:


![](https://i.imgur.com/ErO8woA.png)

The dataset is relatively well-balanced, with the exception of the 'trash' class, which lacks in count compared to the count of the rest of the classes present in the data. 

TODO: show and describe problematic nature of dataset

#### Image Quality for Analysis

pros: minimal visual noise. the pictures contain the item to be classified and only the item to be classified.

cons: lack of consistency with regards to centering, orientation, zoom, and white space. 




![](https://i.imgur.com/NPFauiV.jpg)

these cons present us with a challenge that we welcome with open arms. In production scenarios, an ML agent that is tasked with sorting waste items will likely also face similar orientation and centering issues as it receives image inputs from a conveyor belt. So, working with images of this nature better prepares our ML model to tackle real life waste sorting tasks.  



## Methods




### SVM

A Support Vector Machine (SVM) is a supervised classification algorithm that separates data from different classes by finding optimal hyperplanes between sets of data. The goal of SVM is to maximize the margin between any two separable classes of data. Datapoints that lie on the edge of the hyperplane margin are known as support vectors. 

SVM's can operate on both linearly separable data and non-linearly separable data, depending on which specific 

The three primary questions we answered before finalizing our SVM model:
1. How can we pre-process the image data in a way that enables an SVM to classify the waste item images effectively?
2. How can we split the data into training and testing sets?
3. Should we utilize a linear SVM or a non-linear SVM, and how can we optimize its hyperparameters?

#### Pre-Processing:

In order to make use of SVM effectively for image classification, 

- resize image
- flatten image
- greyscale
- extract hog features
- combine color and hog
- split data
- run that shit
- grid search

the benefits of this preprocessing are two-fold: faster runtime, and less overfitting.

![](https://i.imgur.com/Uc6wGzI.png)


![](https://i.imgur.com/He675EO.png)


#### SVM Classification Report:

![](https://i.imgur.com/DuyeTjh.png)


#### SVM Confusion Matrix:

![](https://i.imgur.com/vYN9Wwl.png =75%x)



one unexpected result that we discovered from our SVM confusion matrix is that there seemed to be a lot of confusion for our model between classifying glass versus classifying trash. Our initial thought was that cardboard and trash would overlap. after examining hog feature images, this result makes more sense. glass and trash are the only two labels whos hog features convey a non-standard polygonal shape. 

todo: show plots of rbf vs linear for diff feature extraction methods
todo: explain why RBF might be better than linear in this case
todo: explain why hog features might be better than pca or batch extraction
todo: explain what gamma means in rbf

![](https://i.imgur.com/zc4Ufjd.png)


example of what flattened image points may look like
red is one class
blue is another class
![](https://i.imgur.com/cyzlcjY.png)

how does rbf kernel svm operate on this kind of datA?



![](https://i.imgur.com/SohAeKT.png)


![](https://i.imgur.com/i3x9A9q.png)

rbf kernel viz sourcE: https://www.youtube.com/watch?v=wuKlhMDxtN0

large gamma = risk of overfitting
small gamma = underfit risk






### Decision Tree and Random Forest

![](https://i.imgur.com/E4tHLZP.png)

#### Decision Tree
Decision trees are a supervised learning method in which data is classified using branching. Different features are analyzed and used to split the data into groups until the majority of the data within a leaf node is data of the same classification. 

    No Feature Extraction
        - Converted the image to gray scale
        - Resized the image
        - Used all the pixels from the resized image for the decision tree
        - Flattened the image
    Accuracy: 0.45011600928074247
![](https://i.imgur.com/G0EyD8d.png =75%x)
![](https://i.imgur.com/Vz0k6n5.png)


    
    Feature Extraction Using Histogram of Oriented Gradients
        - Converted the image to gray scale
        - Resized the image
        - Used HOG on the resized image
        - Flattened the image
    Accuracy: 0.43387470997679817
![](https://i.imgur.com/aI01ObI.png =75%x)
![](https://i.imgur.com/CahDNDR.png)


    
    Feature Extraction Using Harris Corner Detection 
        - Converted the image to gray scale
        - Resized the image
        - Used Harris Corner Detection on the resized image
        - Flattened the image
    Accuracy: 0.45243619489559167
![](https://i.imgur.com/NlaTsYU.png =75%x)
![](https://i.imgur.com/lgL7SmS.png)




#### Random Forest
Random forest is an ensemble algorithm that uses decision trees. Multiple decision trees are created by randomly selecting a subset of data from the dataset. All of these trees are used in the classificaiton of the data in the dataset. 

    No Feature Extraction
        - Converted the image to gray scale
        - Resized the image
        - Used all the pixels from the resized image for the decision tree
        - Flattened the image
    Accuracy: 0.6705336426914154
![](https://i.imgur.com/jse3SHd.png =75%x)

    
    Feature Extraction Using Histogram of Oriented Gradients
        - Converted the image to gray scale
        - Resized the image
        - Used HOG on the resized image
        - Flattened the image
    Accuracy: 0.6496519721577726
![](https://i.imgur.com/QEGqasL.png =75%x)

    
    Feature Extraction Using Harris Corner Detection 
        - Converted the image to gray scale
        - Resized the image
        - Used Harris Corner Detection on the resized image
        - Flattened the image
    Accuracy: 0.6890951276102089
![](https://i.imgur.com/w2UT9n2.png =75%x)




#### Comparision between Decision Tree and Random Forest
Overall, in each of the datasets used, random forest performed better. The reason that random forest performs with higher accuracy is because having multiple trees reduces error from the bias and it also limits overfitting. 


### Neural Network

Building training and testing dataset

* Load images and labels from the directory 
* Resize the image to be in size 128x128
* Choosing a training and testing ratio (In this case we choose 9:1)
* Randomly select the number of training data from each classes according to the ratio
* Shuffle the training and testing data respectively to give more randomness
* Seperate the training and testing data into mini-batches to reduce memory usage (In this case we choose 4)
* Each time feed one batch to the network


#### 1. Multi-layer Perceptron¶
![](https://i.imgur.com/OXVMPKR.png =50%x)
Multi-layer Perceptron(MLP) is a kind of supervised learning algorithm which consists of one input layer, several hidden layers and the output layer. Each hidden layer contains nodes that use nonlinear activation functions. MLP uses backpropagation to train the network and update the nodes. The multi-layer structure and nonlinear activation give MLP the ability to capture more complex and non-linear separable features to classify our data.

In order to feed our training images to the network, we need to flatten our training data(N,C,H,W) to be in the shape of (N,CxHxW). 

Here we compares the loss and accuracy for different network architecture:
* Vary in number of hidden unit

* Vary in number of layers

#### 2. Convolutional Neural Network

Convolutional Neural Network(CNN) is another type of supervise learning that is similar to MLP. In addition to the fully connected layer that MLP has, CNN will first perform convolution on the input image to extract low level features.



## Results 



## Discussion

### team members need to clearly claim their contributions in the project report which is the GitHub page. *** !!

## References

[1]https://medium.com/data-science-bootcamp/multilayer-perceptron-mlp-vs-convolutional-neural-network-in-deep-learning-c890f487a8f1


