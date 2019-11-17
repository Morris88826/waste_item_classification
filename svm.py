import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

# import seaborn as sns
# sns.set()

import pandas as pd
import numpy as np
import pylab as pl

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm, metrics, datasets
from sklearn.utils.multiclass import unique_labels

from sklearn.feature_extraction import image
from skimage.transform import resize

from sklearn.metrics import roc_curve, auc

def get_image(filename, root="garbage-classification\Garbage classification\call"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    # filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    img_resized = resize(np.array(img), (32, 32), mode='reflect')
    # img = np.array(img)
    # # print(img.shape)
    # patches = image.extract_patches_2d(np.array(img), (8, 8), max_patches=8, random_state=0)
    # patches = patches.reshape(64,24)
    # # print(patches.shape)
    # return np.array(patches)
    return np.array(img_resized)

# plt.imshow(get_image('glass26.jpg'))
# plt.show()

# bombus = get_image('glass26.jpg')

# print('Color bombus image has shape: ', bombus)

# # convert the bombus image to greyscale
# grey_bombus = rgb2grey(bombus)

# plt.imshow(grey_bombus, cmap=mpl.cm.gray)
# plt.show()

# print('Greyscale bombus image has shape: ', grey_bombus)

# hog_features, hog_image = hog(grey_bombus,
#                               visualise=True,
#                               block_norm='L2-Hys',
#                               pixels_per_cell=(16, 16))

# plt.imshow(hog_image, cmap=mpl.cm.gray)
# plt.show()


def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack((color_features, hog_features))
    return flat_features

# bombus_features = create_features(bombus)

# print(bombus_features)

imglit = os.listdir("garbage-classification\Garbage classification\call")

alosty = []
for each in imglit:
  if 'cardboard' in each:
    alosty.append('cardboard')
  if 'glass' in each:
    alosty.append('glass')
  if 'metal' in each:
    alosty.append('metal')
  if 'paper' in each:
    alosty.append('paper')
  if 'plastic' in each:
    alosty.append('plastic')
  if 'trash' in each:
    alosty.append('trash')

print('hi')

def create_feature_matrix(imglit):
    features_list = []
    imglist = []
    
    for img_id in imglit:
        # load image
        img = get_image(img_id)
        # print('hi')
        # imglist.append(img)
        # get features for image
        # image_features = create_features(img)
        features_list.append(img)
        # features_list.append(image_features)
        
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix

# run create_feature_matrix on our dataframe of images
feature_matrix = create_feature_matrix(imglit)
# print(feature_matrix.shape)
# feature_matrix = feature_matrix.reshape(1293824, 1152)
# feature_matrix = feature_matrix.reshape(2527, 1536)
feature_matrix = feature_matrix.reshape(2527, 3072)
# feature_matrix = feature_matrix.reshape(782, 3072)

# get shape of feature matrix
print('Feature matrix shape is: ', feature_matrix.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
# bees_stand = ss.fit_transform(feature_matrix)
# print(bees_stand.shape)
# print(bees_stand.shape)



# pca = PCA(n_components=200)
# use fit_transform to run PCA on our standardized matrix
# bees_pca = pca.fit_transform(bees_stand)
bees_pca = ss.fit_transform(feature_matrix)
# look at new shape
print('matrix shape is: ', bees_pca.shape)


X = pd.DataFrame(bees_pca)
y = pd.Series(alosty)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.3,
                                                    random_state=42)

# look at the distrubution of labels in the train set
pd.Series(y_train).value_counts()

# pcaa = PCA(n_components=2).fit(X_train)
# pca_2d = pcaa.transform(X_train)

# for i in range(0, pca_2d.shape[0]):
#   if y_train[i] == 'cardboard':
#     c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    marker='+')
#   elif y_train[i] == 'glass':
#     c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    marker='o')
#   elif y_train[i] == 'plastic':
#     c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    marker='*')
#     pl.legend([c1, c2, c3], ['Setosa', 'Versicolor',    'Virginica'])
#     pl.title('Iris training dataset with 3 classes and    known outcomes')
#     pl.show()

# define support vector classifier
# svm = SVC(kernel='linear', probability=True, random_state=42)
svm = SVC(kernel='rbf', probability=True)

# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
# svc = svm.SVC()
# clf = GridSearchCV(svc, param_grid)
# clf.fit(X_train, y_train)

# scores = [x[1] for x in clf.cv_results_]
# scores = np.array(scores).reshape(4, 2)

# for ind, i in enumerate([1, 10, 100, 1000]):
#     plt.plot([0.001, 0.0001], scores[ind], label='C: ' + str(i))
# plt.legend()
# plt.xlabel('Gamma')
# plt.ylabel('Mean score')
# plt.show()


# fit model
svm.fit(X_train, y_train)

# generate predictions
y_pred = svm.predict(X_test)

class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
labels = [1,2,3,4,5,6]

def generate_confusion_matrix(ground_truth, predicts, labels, cmap= plt.cm.get_cmap('Blues'), normalize=True):
  """
  Args: 
  - ground_truth: The true labels of the image set, shape=(N,1)
  - predicts: The predict labels of the image set, shape=(N,1)
  - labels: The label of your classes
  - cmap: Color map
  - normalize: Normalize the confusion matrix

  Returns:
  - ax
  """
  cm = confusion_matrix(ground_truth, predicts)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')
      
  fig = plt.figure()
  ax = fig.add_subplot(111)

  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(# ... and label them with the respective list entries
          xticklabels=([''] + labels), yticklabels=([''] + labels),
          ylabel='True label',
          xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  plt.show()
  return ax

# calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print('Model accuracy is: ', accuracy)

# ax = generate_confusion_matrix(y_test, y_pred, class_names)



# df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=class_names), range(6),
#                   range(6))
# #plt.figure(figsize = (10,7))
# sn.set(font_scale=1.4)#for label size
# sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size

# plt.show()

# y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)

# print("Classification report for - \n{}:\n{}\n".format(
#    clf, metrics.classification_report(y_test, y_pred)))

# predict probabilities for X_test using predict_proba
# probabilities = svm.predict_proba(X_test)
# print(probabilities)

# # select the probabilities for label 1.0
# y_proba = probabilities[:, 1]
# print(y_proba)

# # calculate false positive rate and true positive rate at different thresholds
# print(roc_curve(y_test, y_proba, pos_label=1))
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# # calculate AUC
# roc_auc = auc(false_positive_rate, true_positive_rate)

# plt.title('Receiver Operating Characteristic')
# # plot the false positive rate on the x axis and the true positive rate on the y axis
# roc_plot = plt.plot(false_positive_rate,
#                     true_positive_rate,
#                     label='AUC = {:0.2f}'.format(roc_auc))

# plt.legend(loc=0)
# plt.plot([0,1], [0,1], ls='--')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate');
# plt.show()


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='Purples'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

plot_classification_report(metrics.classification_report(y_test, y_pred))
plt.show()