import numpy as np

# 
# compute chi2 distance between x and y
#
def dist_chi2(x,y):
  # your code here
  d1 = (x-y)**2
  d2 = x+y
  d = np.zeros(d1.shape)
  # print("############### Shape:",d1.shape)
  for i in range(d1.shape[0]):
    # for j in range(d1.shape[1]):
    if d2[i] != 0:
      d[i] = d1[i]/d2[i]
  d = d.sum()
  return d

# 
# compute l2 distance between x and y
#
def dist_l2(x,y):
  # your code here
  d = x-y
  d = np.multiply(d, d.T)

  return d.sum()

# 
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similart histograms
#
def dist_intersect(x,y):
  # your code here
  d = np.minimum(x,y)
  d = d.sum()
  d = 1-d
  return d

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert 'unknown distance: %s'%dist_name
  




