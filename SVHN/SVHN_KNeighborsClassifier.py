# -*- coding: utf-8 -*-

import numpy as np
import gzip
from sklearn.neighbors import KNeighborsClassifier
import _pickle as cPickle
from PIL import Image
import cv2
from skimage import filters
import tarfile

# load data
f = gzip.open('./mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

x_train = train_set[0]
y_train = train_set[1]

# Create KNeighbors Classifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(x_train, y_train)

def resizeImageTo28X28(im):
  size = 28,28
  imResize = Image.fromarray(np.uint8(im))
  imResize = imResize.resize(size)
  return np.array(imResize)

def paddingImageToSquareImage(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.shape
    size = abs(x-y)
    if size%2==0:
      return np.array([np.pad(row, (int(size/2), int(size/2)), 'constant') for row in im])
    else:
      return np.array([np.pad(row, (int(size/2)+1, int(size/2)), 'constant') for row in im])

def FindingEdges(im):
  im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  sobelx = cv2.Sobel(im, cv2.CV_16S, 1, 0, ksize=3)
  sobely = cv2.Sobel(im, cv2.CV_16S, 0, 1, ksize=3)
  abs_gradientx = cv2.convertScaleAbs(sobelx)
  abs_gradienty = cv2.convertScaleAbs(sobely)

  # combine the two in equal proportions
  total_gradient = cv2.addWeighted(abs_gradientx, 0.5, abs_gradienty, 0.5, 0)
  val = filters.threshold_otsu(total_gradient)
  return np.array(im_gray < val, dtype=np.uint8)

def classfierDigit(im):
  return clf_knn.predict(im)


def classfierHouseNumberImage(im):
    edges_images = FindingEdges(im)
    _,cnts,_  = cv2.findContours(edges_images, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        (x,y,w,h) = cv2.boundingRect(contour)
        print(x,y,w,h)
        # Threshold To ignore out unnecessary edges in an image, you can change but work in most images.
        if h-w > 30 and h-w < 50:
            padding_image = paddingImageToSquareImage(edges_images[y:y + h, x:x + w])
            resize_image = resizeImageTo28X28(padding_image)
            resize_image = resize_image.reshape(-1)
            print(classfierDigit([resize_image]))

test_path = './test_min.tar.gz'
tf = tarfile.open(test_path)
tf.extractall(path='./test')

data_test_path = os.listdir('./test')
test_images = os.listdir(data_test_path)

for image in test_images:
    classfierHouseNumberImage(image)
