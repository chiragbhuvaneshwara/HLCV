import numpy as np
from numpy import histogram as hist
import math
import sys
sys.path.insert(0, '../filter-Q1')

import gauss_module
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    #print("inside normalized")
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    hists =[]
    

    # print(img_gray.shape)
    img_gray = img_gray.flatten()
    sizeOfEachBin = 255/num_bins
    start = 0.0
    end = float(sizeOfEachBin)
    bins = [start]

    while end <= 255:
        
      count = 0
      for grayVal in img_gray:
        if grayVal <= end and grayVal >= start:
          count += 1 
      hists.append(count)
      bins.append(end)
      start += sizeOfEachBin
      end += sizeOfEachBin
      

    hists = np.array(hists)
    total = hists.sum()
    hists = hists / total

    bins = np.array(bins)
    #print(len(hists))

    return hists,bins


#  compute joint histogram for each color channel in the image, histogram should be normalized so that sum of all values equals 1
#  assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
def rgb_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    
    # execute the loop for each pixel in the image 
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            sizeOfEachBin = 256/num_bins
            R = math.floor(img_color[i,j,0]/sizeOfEachBin) 
            G = math.floor(img_color[i,j,1]/sizeOfEachBin) 
            B = math.floor(img_color[i,j,2]/sizeOfEachBin) 
            hists[R,G,B] += 1

    # normalize the histogram such that its integral (sum) is equal 1
    # your code here

    hists = hists.reshape(hists.size)
    total = hists.sum()
    hists = hists / total

    return hists

#  compute joint histogram for r/g values
#  note that r/g values should be in the range [0, 1];
#  histogram should be normalized so that sum of all values equals 1
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
def rg_hist(img_color, num_bins):

    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'
  
    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    t = 1/(num_bins -1)
    
    # your code here
    for i in range(img_color.shape[0]):
      for j in range(img_color.shape[1]):
        denom = img_color[i,j,0]+img_color[i,j,1]+img_color[i,j,2]
        r = img_color[i,j,0]/denom
        g = img_color[i,j,1]/denom

        r = int(r/t)
        g = int(g/t)

        if r > num_bins:
          r = num_bins
        elif r < 1:
          r = 1
        if g > num_bins:
          g = num_bins
        elif g < 1:
          g = 1
        
        hists[r,g] += 1

   # hists = hists/(img_color.shape[0]*img_color.shape[1])
    hists = hists.reshape(hists.size)
    total = hists.sum()
    hists = hists / total
    #hists = hists.reshape(hists.size)
    return hists


#  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
#  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
#  histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input grayvalue image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  note: you can use the function gaussderiv.m from the filter exercise.
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # compute the first derivatives
    sigma = 7.0
    img_dx, img_dy = gauss_module.gaussderiv(img_gray, sigma)

    # quantize derivatives to "num_bins" number of values
    min_a = -32
    max_a = 32

    t = (max_a - min_a + 1)/num_bins

    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    # ...
    for i in range(img_gray.shape[0]):
      for j in range(img_gray.shape[1]):
        x = math.floor((img_dx[i,j]+32)/t)
        y = math.floor((img_dy[i,j]+32)/t)
        hists[x,y] += 1
    
    hists = hists.reshape(hists.size)
    hists = hists / (img_gray.shape[0]*img_gray.shape[1])
    total = hists.sum()
    hists = hists / total

    return hists

def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
  if dist_name == 'grayvalue':
    hist, _ = normalized_hist(img1_gray, num_bins_gray)
    return hist
  elif dist_name == 'rgb':
    return rgb_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rg':
    return rg_hist(img1_gray, num_bins_gray)
  elif dist_name == 'dxdy':
    return dxdy_hist(img1_gray, num_bins_gray)
  else:
    assert 'unknown distance: %s'%dist_name
  
