import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#from skimage.color import rgb2gray
#import matplotlib.image as mpimg
#import heapq as hp

import histogram_module
import dist_module
from dist_module import *
from histogram_module import *

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#
# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image
#

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

  hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

  model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
  query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

  D = np.zeros((len(model_images), len(query_images)))
  
  best_match = []
  # your code here
  for i in range(len(model_images)):
    for j in range(len(query_images)):
      D[i,j] = get_dist_by_name(model_hists[i], query_hists[j], dist_type)

  for i in range(len(query_images)):
    best_match.append(np.argmin(D[:,i]))
  best_match = np.array(best_match)

  return best_match, D

def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

  image_hist = []

  # compute hisgoram for each image and add it at the bottom of image_hist
  # your code here
  for i in range(len(image_list)):
    #img_color = imread(char(string(image_list(i))))
    img_color = np.array(Image.open(image_list[i]))
    if(hist_isgray):
      img = rgb2gray(img_color.astype('double'))
    else:
      img = img_color.astype('double')
  
    #image_hist.append(get_hist_by_name(img_color.astype('double'),num_bins,hist_type))
    image_hist.append(get_hist_by_name(img,num_bins,hist_type))


  return image_hist

#
# for each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# note: use the previously implemented function 'find_best_match'
# note: use subplot command to show all the images in the same Python figure, one row per query image
#

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

  #plt.figure()
  num_nearest = 5  # show the top-5 neighbors

  # your code here
  [best_match, D] = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

  for i in range(len(query_images)):
    plt.figure()
    idx = np.argsort(D[:,i])
    idx = idx[:num_nearest]
    plt.subplot(1,6,1) 
    plt.imshow(np.array(Image.open(query_images[i])), vmin=0, vmax=255) 
    plt.title('Query Image')
    for j in range(len(idx)):   
      plt.subplot(1,6,j+2) 
      plt.imshow(np.array(Image.open(model_images[idx[j]])), vmin=0, vmax=255)
    plt.title('Matched Image')
    plt.show()


    

 


