import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple
from matplotlib.pyplot import imshow
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def load_image(path: str) -> np.ndarray:
  """
  Args:
  - path: string representing a file path to an image

  Returns:
  - float or double array of shape (m,n,c) or (m,n) and in range [0,1],
    representing an RGB image
  """
  img = PIL.Image.open(path)
  img = np.asarray(img)
  float_img_rgb = im2single(img)
  return float_img_rgb

def im2single(im: np.ndarray) -> np.ndarray:
  """
  Args:
  - img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]

  Returns:
  - im: float or double array of identical shape and in range [0,1]
  """
  im = im.astype(np.float32) / 255
  return im

def numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: False) -> PIL.Image:
  """
  Args: 
  - img: in [0,1]

  Returns:
  - img in [0,255]
  """
  _img = np.copy(img)
  if scale_to_255:
    _img *= 255
  return PIL.Image.fromarray(np.uint8(_img))

def PIL_image_to_numpy_arr(img, downscale_by_255=True):
  """
  Args:
  - img
  - downscale_by_255

  Returns:
  - img
  """
  img = np.asarray(img)
  img = img.astype(np.float32)
  _img = np.copy(img)
  if downscale_by_255:
    _img = _img/255
  return _img

def PIL_resize(img: np.ndarray, ratio:Tuple[float, float]) -> np.ndarray:
  """
  Args:
  - img: Array representing an image
  - size: Tuple representing new desired (width, height)

  Returns:
  - img
  """
  H, W, _ = img.shape
  img = numpy_arr_to_PIL_image(img, scale_to_255=True)
  img = img.resize((int(W*ratio[1]), int(H*ratio[0])), Image.ANTIALIAS)
  img = PIL_image_to_numpy_arr(img)
  return img

def PIL_resize_ws(img: np.ndarray, ws: int) -> np.ndarray:
  img = numpy_arr_to_PIL_image(img, scale_to_255=True)
  img = img.resize((ws, ws), Image.ANTIALIAS)
  img = PIL_image_to_numpy_arr(img)
  return img

def show_image(img: np.ndarray) -> PIL.Image:
  """
  Args: 
  - img: in [0,1]
  """
  i_img = np.copy(img)
  i_img *= 255
  PIL.Image.fromarray(np.uint8(i_img)).show()
  return


def generate_confusion_matrix(ground_truth, predicts, labels, cmap= plt.cm.get_cmap('Blues'), normalize=False):
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
  return ax