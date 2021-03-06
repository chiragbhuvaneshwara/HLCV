import numpy as np
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module

# 
# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image
#
def plot_rpc(D, plot_color):
  
  recall = []
  precision = []
  total_imgs = D.shape[1]

  num_images = D.shape[0]
  assert(D.shape[0] == D.shape[1])

  labels = np.diag([1]*num_images)
  
  d = D.reshape(D.size)
  l = labels.reshape(labels.size)
 
  sortidx = d.argsort()
  d = d[sortidx]
  l = l[sortidx]
  min_d = np.amin(d)
  max_d = np.amax(d)      #check syntax

  for z in range(len(d)):
    tp = 0
    fp = 0
    fn = 0
    thresh = d[z]
    for idx in range(len(d)):
      if l[idx] == 0 and d[idx] <= thresh:
        fp = fp + 1
      elif l[idx] == 1 and d[idx] <= thresh:
        tp = tp + 1
      elif l[idx] == 1 and d[idx] > thresh:
        fn = fn + 1
    # compute precision and recall values and append them to "recall" and "precision" vectors
      # your code here
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    precision.append(prec)
    recall.append(rec)
  #print(precision, recall)
  plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')


def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):

  assert len(plot_colors) == len(dist_types)

  for idx in range( len(dist_types) ):

    [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)
    
    
    plot_rpc(D, plot_colors[idx])
    
    

  plt.axis([0, 1, 0, 1])
  plt.xlabel('1 - precision')
  plt.ylabel('recall')

  # legend(dist_types, 'Location', 'Best')

  plt.legend( dist_types, loc='best')
  




