# Waste Item Classification
**Mu-Ruei Tseng**, **Tara Poteat**, **Ankit Verma**


## Introduction
Managing waste efficiently is a significant challenge. In 2015, 262 million tons of solid waste were processed in the United States alone. 91 million tons of this waste was recycled[1]. Clearly, recycling plays a large role in waste management. The most prominent recycling system in use currently is the Single Stream Recycling System, which involves manual item sorting by humans[2]. However, classifying waste manually is both time inefficient and space inefficient. This leads us to believe that machines will soon have a greater responsibility in classifying waste items. Through this project, we will develop a model that leverages machine learning algorithms to identify an item’s ‘waste-type’ based on an input image.

![](https://i.imgur.com/BVPqZ6f.png =70%x) 
[3]


![](https://i.imgur.com/cqw98ou.png =70%x) 
[4]







## Dataset


The dataset we used came from Kaggle [5]. The dataset is composed of 2527 images of waste items that can each be classified as either cardboard, glass, metal, paper, plastic, or trash.

Text files containing labels corresponding to each image filename are also present in this dataset. The dataset's count of images by class label is shown below:


![](https://i.imgur.com/ErO8woA.png)

The dataset is relatively well-balanced, with the exception of the 'trash' class, which lacks in count compared to the count of the rest of the classes present in the data. A balanced distribution helps when training ML models. An ML model trained with evenly distributed labelled data is more likely to produce accurate testing results. 

#### Preprocessing the Data
Although there were many images within the dataset and the dataset contained relatively balanced data, one issue that came across was extracting fearures. Besides the images, there was no other data or information about the items that were being classified. This made preparing our data for training difficult because it forced us to find different methods to extract our own features in order to use the SVM and decision tree algorithms. 

We tried multiple extraction techniques on the pixels and some worked better than others. The first way in which we got features from the images was downsizing the images and using the greyscale pixels. Another technique was histogram of oriented gradients which uses the orientation and intensity of edge directions within an image. The last technique that we looked at was Harris Corner Detector which finds locations in the image where two edges meet. 


#### Image Quality for Analysis

The images in the dataset had some ideal characteristics but they also had some more challenging aspects that caused difficulties in the learning process. One of the pros was that the images had minimal visual noise. The pictures contain the item to be classified and only the item to be classified so it wasn't cluttered with random objects. However, a con of the dataset was that it lacked consistency with regards to the centering, orientation, zoom, and white space of each image.




![](https://i.imgur.com/NPFauiV.jpg)

The cons present us with a challenge that we welcome with open arms. In production scenarios, an ML agent that is tasked with sorting waste items will likely also face similar orientation and centering issues as it receives image inputs from a conveyor belt. So, working with images of this nature better prepares our ML model to tackle real life waste sorting tasks.  



## Methods


Our approach involved training four different supervised learning methods, optimizing their hyperparameters, and comparing their prediction accuracies against each other in order to determine which model performs best in waste item classification.

The four supervised learning methods we considered are:

1. KNN 
2. SVM
3. Decision Trees & Random Forests
4. Neural Networks


### KNN
K-Nearest-Neighbors(KNN) is a supervised learning algorithm. It determines the label of the testing data by the dominant label of the K shortest distance between it with other training data.


#### *Naive KNN*
Here we try an easy way of using KNN. Since our original images are too big, we first converted the RGB images into grayscale and resized it be 32x32. We then flatten the image to 1D and calculate its 6 nearest neighbor(since we have 6 different classes). We classified the test data to be the dominant label among the closest 6.

Here is the result:
![](https://i.imgur.com/Kv8WtcP.png =50%x)

Accuracy = 0.32

#### *KNN with SIFT net*
In the second way, we try to first use the SIFT descriptor to get features and then run k-means clusters to separate the features into k regions and save the centers. Whenever we get new SIFT features, we can classify which region it belongs using KNN. 

Here we extract 50 features and this is the result:
![](https://i.imgur.com/2OSJ1qU.png =50%x)
Accuracy = 0.266

### SVM

A Support Vector Machine (SVM) is a supervised classification algorithm that separates data from different classes by finding optimal hyperplanes between sets of data. The goal of SVM is to maximize the margin between any two separable classes of data. Datapoints that lie on the edge of the hyperplane margin are known as support vectors. 

SVM's can operate on both linearly separable data and non-linearly separable data, depending on which specific 

The three primary questions we answered before finalizing our SVM model:
1. How can we pre-process the image data in a way that enables an SVM to classify the waste item images effectively?
2. How can we split the data into training and testing sets?
3. Should we utilize a linear SVM or a non-linear SVM, and how can we optimize its hyperparameters?

#### Pre-Processing:

In order to make use of SVM effectively for image classification, The following steps were taken to pre-process images in the dataset:

- Resized images to 128x128 square images
- Flattened image matrices to one dimension 
- Converted images to greyscale
- Extracted HOG features from greyscale data
- Stacked together flattened image data with HOG features

![](https://i.imgur.com/Uc6wGzI.png)


![](https://i.imgur.com/He675EO.png)


The above preprocessing steps enabled reduction in overfitting while training SVM's, and also significantly improved the runtime of our SVM model.  


#### Benefits of Guassian RBF kernel SVM:

Example of what flattened image points may look like:

![](https://i.imgur.com/cyzlcjY.png=90%x)


How does RBF kernel SVM operate on this kind of data?



![](https://i.imgur.com/SohAeKT.png =90%x)




#### Grid Search:

GridSearchCV was utilized to optimize the SVM's hyperparameters. Grid Search allowed us to measure the performance of RBF vs Linear SVM as well as test out several values of gamma. Gamma is an important parameter since a large gamma value causes a risk of overfitting while a small gamma value causes a risk of underfitting.

![](https://i.imgur.com/i3x9A9q.png =90%x)

[6]


#### SVM Classification Report:

![](https://i.imgur.com/DuyeTjh.png)


#### SVM Confusion Matrix:

![](https://i.imgur.com/vYN9Wwl.png =75%x)



One unexpected result that we discovered from our SVM confusion matrix is that there seemed to be a lot of confusion for our model between classifying glass versus classifying trash. Our initial thought was that cardboard and trash would overlap. after examining hog feature images, this result makes more sense. glass and trash are the only two labels whos hog features convey a non-standard polygonal shape. 

#### Future Improvements

To train this SVM model, a relatively basic call was made to sklearn's 'train_test_split' with 70% training and 30% testing. Perhaps our SVM could be optimized even further by utilizing k-fold cross validation in the future. 








### Decision Tree and Random Forest

![](https://i.imgur.com/E4tHLZP.png)

#### Decision Tree
Decision trees are a supervised learning method in which data is classified using branching. Different features are analyzed and used to split the data into groups until the majority of the data within a leaf node is data of the same classification. 

##### ***No Feature Extraction***
 - Converted the image to gray scale
 - Resized the image
 - Used all the pixels from the resized image for the decision tree
 - Flattened the image
 - Accuracy: 0.450

![](https://i.imgur.com/G0EyD8d.png =75%x)
![](https://i.imgur.com/Vz0k6n5.png)


##### ***Feature Extraction Using Histogram of Oriented Gradients***
 - Converted the image to gray scale
 - Resized the image
 - Used HOG on the resized image
 - Flattened the image
 - Accuracy: 0.434
    
![](https://i.imgur.com/aI01ObI.png =75%x)
![](https://i.imgur.com/CahDNDR.png)


    
##### ***Feature Extraction Using Harris Corner Detection ***
 - Converted the image to gray scale
 - Resized the image
 - Used Harris Corner Detection on the resized image
 - Flattened the image
 - Accuracy: 0.452
    
![](https://i.imgur.com/NlaTsYU.png =75%x)
![](https://i.imgur.com/lgL7SmS.png)




#### Random Forest
Random forest is an ensemble algorithm that uses decision trees. Multiple decision trees are created by randomly selecting a subset of data from the dataset. All of these trees are used in the classificaiton of the data in the dataset. 

##### ***No Feature Extraction***
 - Converted the image to gray scale
 - Resized the image
 - Used all the pixels from the resized image for the decision tree
 - Flattened the image
 - Accuracy: 0.671
![](https://i.imgur.com/jse3SHd.png =75%x)

    
##### ***Feature Extraction Using Histogram of Oriented Gradients***
 - Converted the image to gray scale
 - Resized the image
 - Used HOG on the resized image
 - Flattened the image
 - Accuracy: 0.650
![](https://i.imgur.com/QEGqasL.png =75%x)

    
##### ***Feature Extraction Using Harris Corner Detection***
 - Converted the image to gray scale
 - Resized the image
 - Used Harris Corner Detection on the resized image
 - Flattened the image
 - Accuracy: 0.689
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


#### Multi-layer Perceptron¶
![](https://i.imgur.com/OXVMPKR.png =50%x)
Multi-layer Perceptron(MLP) is a kind of supervised learning algorithm which consists of one input layer, several hidden layers and the output layer. Each hidden layer contains nodes that use nonlinear activation functions. MLP uses backpropagation to train the network and update the nodes. The multi-layer structure and nonlinear activation give MLP the ability to capture more complex and non-linear separable features to classify our data.

In order to feed our training images to the network, we need to flatten our training data(N,C,H,W) to be in the shape of (N,CxHxW). 

Here we compares the loss and accuracy for different network architecture:


##### ***Vary in number of hidden units***

Since we do not know the amount of hidden units that best fit our model, we tried several numbers and found the network with the highest accuracy. We tested it with 9 kinds of hidden units. Here is the test result and the confusion map for each architecture.

20 hidden units |30 hidden units             |  40 hidden units
:-------------------------:|:-------------------------:|:-------------------------:
 ![](https://i.imgur.com/QOYxWZV.png) | ![](https://i.imgur.com/cHTH6gQ.png) | ![](https://i.imgur.com/OBqGz8z.png)



50 hidden units |60 hidden units             |  70 hidden units
:-------------------------:|:-------------------------:|:-------------------------:
![](https://i.imgur.com/osT6PNx.png) | ![](https://i.imgur.com/f7zJfU5.png)| ![](https://i.imgur.com/PYLRa45.png)



80 hidden units |90 hidden units             |  100 hidden units
:-------------------------:|:-------------------------:|:-------------------------:
![](https://i.imgur.com/P767bW3.png)| ![](https://i.imgur.com/1KlsqFR.png) | ![](https://i.imgur.com/782FyE1.png)

Loss |Accuracy             | 
:-------------------------:|:-------------------------:
![](https://i.imgur.com/B6nrDLo.png)|![](https://i.imgur.com/IL0Foh8.png)

Losses: [0.688, 0.565, 0.570, 0.695, 0.669, 0.571, 0.666, 0.550, 0.527]
Accuracies: [0.461, 0.465, 0.465, 0.426, 0.480, 0.453, 0.434, 0.480, 0.520]



##### ***Vary in number of layers***

In this part, we use the number of hidden nodes that gives out the best result and apply it with more hidden layers. 
The number of hidden units we chose to apply is 100 hidden units and we tested it with 4 hidden layers. Here are the results.

1 hidden layer |2 hidden layers            |  3 hidden layers  | 4 hidden layers 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://i.imgur.com/wRvKANm.png) | ![](https://i.imgur.com/FSwjog8.png) | ![](https://i.imgur.com/1FMRWUY.png) | ![](https://i.imgur.com/jxYnkfJ.png)

Loss |Accuracy             | 
:-------------------------:|:-------------------------:
![](https://i.imgur.com/EwtkUJR.png)|![](https://i.imgur.com/oSFE6EY.png)



Losses: [0.594, 0.717, 0.872, 0.618]
Accuracies: [0.453, 0.516, 0.527, 0.559]



#### Convolutional Neural Network

Convolutional Neural Network(CNN) is another type of supervise learning that is similar to MLP. In addition to the fully connected layer that MLP has, CNN will first perform convolution on the input image to extract low level features. Here is our network structure.

##### ***Network 1***
![](https://i.imgur.com/r4yZTZz.png)

![](https://i.imgur.com/wyTF8K3.png)

Loss and accuracy:
Here we show the loss, convolution and accuracy for different epochs.

Loss |Convolution Map             | 
:-------------------------:|:-------------------------:
![](https://i.imgur.com/hiSO6ff.png)|![](https://i.imgur.com/uqXTnWf.png)

Accuracy: 0.381

##### ***Network 2***
![](https://i.imgur.com/aggp4Q0.png)
![](https://i.imgur.com/0zsEtZC.png)

Loss and accuracy:
Here we show the loss, convolution and accuracy for different epochs.

Loss |Convolution Map             | 
:-------------------------:|:-------------------------:
![](https://i.imgur.com/ki2Bdv7.png)|![](https://i.imgur.com/9gGu3Sx.png)

Accuracy: 0.278


## Results
Overall, the results of the data through using the different models was decent. Some of the algorithms and modifications performed better than others, but with K-Nearest-Neighbors, SVM, decision tree and random forest, and neural networks, the accuracy ranged from 25% to 70%. Using neive KNN produced an accuracy of 32% and the modified KNN with extracted SIFT features produced an accuracy of 26.6%. It was surprising that the SIFT KNN was not as accurate as the nieve because with SIFT, features were intentionally extracted. The next method was SVM. With SVM we found that the accuracy was close to 63%. Next we trained the data using decsion treee and random forest, we found that the most accurate decison tree had an accuracy of around 45% and random forest had an accuracy of around 69%. The random forest was over 20% more accurate than decsion trees, which makes sense because random forest uses decision trees and is intended to improve them. Lastly with the neural networks we used both a multilayer perceptron and convolutional neural network, each with a varying number of hidden layers. As the number of hidden layers increased, the accuracy increased. In the multilayer perceptron network, the highest accuracy was around 56% and the convolutional neural network did not performa as well with aroudn 38% accuracy.

Below are comparisons of the accuracy results of the different methods that we used: 

![](https://i.imgur.com/3Agg6o3.png =70%x)


    KNN-1: Naive KNN
    KNN-2: KNN with SIFT net
    SVM
    DT-1: No Feature Extraction
    DT-2: Feature Extraction Using Histogram of Oriented Gradients
    DT-3: Feature Extraction Using Harris Corner Detection  
    RF-1: No Feature Extraction
    RF-2: Feature Extraction Using Histogram of Oriented Gradients
    RF-3: Feature Extraction Using Harris Corner Detection  
    MLP-1: Vary in number of hidden units
    MLP-2: Vary in number of layers
    CNN-1: One convolutional layer
    CNN-2: Two convolutional layers

## Discussion
With our results, the most accurate algorithm was random forest. The next highest accuracy was from SVM and then the mutilayer perceptron network. These results were not as accurate as we had hoped originally, but the images were not as consistent in the location, orientation, zoom, and white space of the item. Because of these challenges, there are further improvements we can make. Based on our results and findings, we can use these more accurate base models that we have produced to further improve the training of the data. We can do this by finding more ideal weights and feature extraction methods to make the models more accurate than they already are. Additionally, if further developed, the waste item classification model that results from our project can prove to be of value to waste management facilities that engage in automated sorting. Our project is limited in scope since we are identifying images of a confined format containing singular items. For a real-world use case, analyzing a photo for each and every trash item would likely be slow and cost-ineffective. We can further improve upon our proposed model by training it to be able to classify types of waste in pictures containing many items at once. This development would significantly shorten the amount of time it would take to sort the items, and prove more cost-effective to waste management facilities. 



## Contribution
SVM Classification: Ankit Verma
Decision Tree and Random Forest Classification: Tara Poteat
K-Means and Neural Network Classifcation: Mu-Ruei Tseng
Write-up: Ankit Verma, Tara Poteat, Mu-Ruei Tseng


## References

[1 ] US EPA. (2018, June). Advancing Sustainable Materials Management: 2015 Fact Sheet. Retrieved from: https://www.epa.gov/sites/production/files/2018-07/documents/2015_smm_msw_factsheet_07242018_fnl_508_002.pdf

[2] LeBlanc, R. (2019, June 25). Single Stream Recycling Offers Benefits, Creates Challenges. Retrieved October 1, 2019, from https://www.thebalancesmb.com/an-overview-of-single-stream-recycling-2877728.

[3] Labor Turnover Examined in SW Missouri. (n.d.). Retrieved from http://www.workforcezone.net/turnover/.http://www.workforcezone.net/turnover/.

[4] Trends in Solid Waste Management. (2019). Retrieved from http://datatopics.worldbank.org/what-a-waste/trends_in_solid_waste_management.html.

[5] Cchangcs. (2018, November 24). Garbage classification. Retrieved from https://www.kaggle.com/asdasdasasdas/garbage-classification.

[6] Support Vector Machines Radial Basis Function Kernel 3. (n.d.). Retrieved from https://www.youtube.com/watch?v=wuKlhMDxtN0.

[7]Uniqtech. (2019, June 13). Multilayer Perceptron (MLP) vs Convolutional Neural Network in Deep Learning. Retrieved from https://medium.com/data-science-bootcamp/multilayer-perceptron-mlp-vs-convolutional-neural-network-in-deep-learning-c890f487a8f1.


<p align="center">
  <img width="460" height="300" src="http://www.fillmurray.com/460/300">
</p>