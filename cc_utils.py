import numpy as np
import pandas as pd
import cv2
from numpy import genfromtxt
from sklearn.cross_decomposition import PLSRegression

# def ApplyRGBTransform(rgb,BT):
#     # faster method
#     B = BT[0]
#     result = rgb + B
#     return result

def ApplyRGBTransform(rgb,BT):
    # faster method
    B = BT[0]
    M = BT[1]
    rgb_reshaped = rgb.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))
    result = np.dot(M, rgb_reshaped.T).T.reshape(rgb.shape)
    result = result + B
    result = np.clip(result,0,255)
    return result

def ApplyRGBTransformPLS(org_im,pls):
    # faster method
    calib_img = org_im.copy()
    for im in calib_img:
        im[:] = pls.predict(im[:])
    return calib_img


def FindRGBTransformPLS(rgbFrom, rgbTo):
    pls = PLSRegression(n_components=3)
    pls.fit(rgbFrom, rgbTo)
    sc = pls.score(rgbFrom, rgbTo)
    print(sc)
    return pls

def FindRGBTransform(rgbFrom, rgbTo, bias = True, transf = False):
    #TODO
  #  rgbFrom = rgbFrom[:,::-1]


    B = [0,0,0]
    T = np.eye(3)

    if(bias):
        mFrom = np.mean(rgbFrom,0)
        mTo = np.mean(rgbTo, 0)
        B = mTo - mFrom
        B = B - np.mean(B)#TEMP
    if(transf):
        rgbFrom_centered = rgbFrom - np.mean(rgbFrom, 0)
        rgbTo_centered = rgbTo - np.mean(rgbTo, 0)
        temp = np.linalg.pinv(rgbFrom_centered)
        T = np.dot(temp, rgbTo_centered)
        B = np.mean(rgbTo, 0) - np.dot(np.mean(rgbFrom, 0),T)



    return [B,T]


def FindRGBTransformWLS(rgbFrom, rgbTo, bias = True, transf = False):
    #TODO

    w = np.ones(24, dtype=float)
    w[18:] = 4.0 # gray values
    w[18] = 10.0 # white is most important

    nw = w / w.sum()  # normalized
    NW = np.diag(nw)
    W = np.sqrt(np.diag(nw))

    B = [0,0,0]
    T = np.eye(3)

    if(bias):
        mFrom = np.sum(np.dot(NW,rgbFrom),0)
        mTo = np.sum(np.dot(NW,rgbTo),0)
        B = mTo - mFrom
    if(transf and bias):

        rgbFrom_centered_ww = rgbFrom - np.sum(np.dot(NW,rgbFrom),0)
        rgbTo_centered_ww = rgbTo - np.sum(np.dot(NW,rgbTo),0)
        # temp = np.linalg.pinv(rgbFrom_centered)
        # T = np.dot(temp, rgbTo_centered)
        W = np.sqrt(np.diag(w))
        Aw = np.dot(W, rgbFrom_centered_ww)
        Bw = np.dot(W,rgbTo_centered_ww)
        TT = np.linalg.lstsq(Aw, Bw)
        T = TT[0]
        B = np.sum(np.dot(NW,rgbTo),0) - np.dot(np.sum(np.dot(NW,rgbFrom),0),T)
    if (transf and not bias):
        rgbFrom_centered = rgbFrom - np.mean(rgbFrom, 0)
        rgbTo_centered = rgbTo - np.mean(rgbTo, 0)
        # temp = np.linalg.pinv(rgbFrom_centered)
        # T = np.dot(temp, rgbTo_centered)

        Aw = np.dot(W, rgbFrom)
        Bw = np.dot(W, rgbTo)
        TT = np.linalg.lstsq(Aw, Bw)
        T = TT[0]




    return [B,T]


def resizeTo(im, w):
	rsz = w/im.shape[1]
	rr = int(im.shape[0]*rsz)
	cc = int(im.shape[1]*rsz)
	im1 = cv2.resize(im,(cc, rr), interpolation= cv2.INTER_LANCZOS4)
	return im1, rsz


def GetRGBs(im, ptrs, nei = 5):
    N = len(ptrs)
    rgbs = []
    for p in ptrs:
        p = p[::-1]
        cc = im[p[0] - nei:p[0] + nei, p[1] - nei:p[1] + nei, :]
        b = np.median(cc[:,:,0])
        g = np.median(cc[:,:,1])
        r = np.median(cc[:,:,2])
        rgbs.append([r,g,b])
    return rgbs


def GetTemplate(ind = 0):
    my_data = genfromtxt('template/template.csv', delimiter=',')
    if(ind == 10):
        my_data = genfromtxt('template/template10.csv', delimiter=',')
    if (ind == 2):
        my_data = genfromtxt('template/template_sun24.csv', delimiter=',')
    if (ind == 34):
        my_data = genfromtxt('template/template34.csv', delimiter=',')
    my_data = my_data[:,::-1]
    return my_data