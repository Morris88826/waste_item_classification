import torch
import numpy as np
import PIL
from PIL import Image
from typing import Tuple
from matplotlib.pyplot import imshow

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

def show_image(img: np.ndarray) -> PIL.Image:
  """
  Args: 
  - img: in [0,1]
  """
  i_img = np.copy(img)
  i_img *= 255
  PIL.Image.fromarray(np.uint8(i_img)).show()
  return