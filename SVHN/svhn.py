# -*- coding: utf-8 -*-

import cv2

from PIL import Image

import gzip

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
import imutils
import tarfile
import os

def trainPca():
    f = open("./mnist_train_small.csv", 'r')
    data = f.readlines()
    f.close()

    train_images = []
    train_labels = []
    for line in data:
        linebits = line.split(',')
        train_images.append(linebits[1:])
        train_labels.append(linebits[0])

    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_images)
    # Apply transform to both the training set and the test set.
    train_images = np.array(scaler.transform(train_images))

    digits = []
    # layers for traning for each digit
    for i in range(10):
        digits.append([])
        digits_true = [int(label) == i for label in train_labels]

        for j in range(len(digits_true)):
            if digits_true[j]==True:
              digits[i].append(train_images[j])

    print(max(len(digits[i]) for i in range(len(digits))))

    digits_pca = []
    for i in range(10):
        digits_pca.append(PCA(.95))
        digits_pca[i].fit(digits[i])

    return digits_pca

def classfierDigit(test_im):
  digits_pca = trainPca()
  score = []
      #transform and inverse
  for i in range(10):
      array = np.array(test_im)
      array = array.reshape(784)
      test_tran = digits_pca[i].transform([array])
      inverse_trans = digits_pca[i].inverse_transform(test_tran).reshape([28,28])
      #Euclidean distance
      score.append(np.linalg.norm(array - inverse_trans.ravel()))
      #For a model of a single image review
  theModule = score.index(min(score))
  return (theModule)

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image

def resizeTo28(image):
  size = 28,28
  im = Image.fromarray(np.uint8(image))
  im = im.resize(size)
  return np.array(im)

def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('gray', (size, size), fill_color)
    new_im.paste(im, ((size - x) / 2, (size - y) / 2))
    return new_im

def FindingEdges(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(im_gray, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(im_gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_gradientx = cv2.convertScaleAbs(sobelx)
    abs_gradienty = cv2.convertScaleAbs(sobely)

    # combine the two in equal proportions
    return cv2.addWeighted(abs_gradientx, 0.5, abs_gradienty, 0.5, 0)

def SearchDigitAndCheck(im):
    im = FindingEdges(im)
    h,w = im.shape
    (winH, winW) = (int(h/2), int(w/4))
    size = 28,28

    for resized in pyramid(im, scale=1.5):
      for (x, y, window) in sliding_window(resized, stepSize=25, windowSize=(winW, winH)):
          # if the window does not meet our desired window size, ignore it
          if window.shape[0] != winH or window.shape[1] != winW:
            continue
    #       if(mone<20):
          im = Image.fromarray(np.uint8(window))
          im = im.resize(size)
          window = np.array(im)
          print(classfierDigit(window))

test_path = './test_min.tar.gz'
tf = tarfile.open(test_path)
tf.extractall(path='./test')

data_test_path = os.listdir('./test')
test_images = os.listdir(data_test_path)

for image in test_images:
    SearchDigitAndCheck(image)