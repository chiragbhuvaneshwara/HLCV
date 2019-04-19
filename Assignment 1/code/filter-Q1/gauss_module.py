# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math

p = math.pi

def gauss(sigma):
    sigma = int(sigma)
    x= [i for i in range(-3*sigma,3*sigma+1, 1)]
        
    Gx = []
    for j in x:
        Gx.append( (1/(sigma*math.sqrt(2*p))) * math.exp(-(j**2/(2 * sigma**2)) ) )

    Gx = np.asarray(Gx)
    x = np.asarray(x)

    return Gx, x

def gaussderiv(img, sigma):
    D , _ = gaussdx(sigma)
    
    imgDx = convAlongRows(img, D)
    imgDy = convAlongCols(img, D)

    return imgDx, imgDy

def gaussdx(sigma):
    sigma = int(sigma)
    x = [i for i in range(-3*sigma, 3*sigma +1, 1)]

    D = []
    for j in x:
        D.append( -(1/(sigma**3 *math.sqrt(2*p)))*(math.exp(-j**2/(2*sigma**2)))*j )

    D = np.asarray(D)
    x = np.asarray(x)

    return D, x

def convAlongRows(img, g):

    allRows = []
    
    for row in img:
        allRows.append(np.convolve(row, g, "same"))

    xConvImg = np.asarray(allRows)

    return xConvImg

def convAlongCols(xConvImg, g):
    transImg = xConvImg.transpose()

    yConvImgTrans = convAlongRows(transImg, g)

    yConvImg = yConvImgTrans.transpose()

    return yConvImg

def gaussianfilter(img, sigma):
    # copyImg = img = [:]
    g, _ = gauss(sigma)
    
    xConvImg = convAlongRows(img, g)
    
    outimage = convAlongCols(xConvImg, g)

    return outimage
    
