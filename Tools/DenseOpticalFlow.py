# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:54:48 2018

@author: HP_OWNER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import time
import math
import glob
import os

def imsave(image, path):
    
    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(path, image*1.0)

# Warps first frame into second frame
def denseFlowMc(frame1, frame2, saveDir):
    
        
        flow0 = cv2.calcOpticalFlowFarneback(frame1[...,0],frame2[...,0], None, 0.5, 6, 10, 6, 7, 1.5, 0).astype(np.float32)
        flow1 = cv2.calcOpticalFlowFarneback(frame1[...,1],frame2[...,1], None, 0.5, 6, 10, 6, 7, 1.5, 0).astype(np.float32)
        flow2 = cv2.calcOpticalFlowFarneback(frame1[...,2],frame2[...,2], None, 0.5, 6, 10, 6, 7, 1.5, 0).astype(np.float32)
        
        '''img = np.dstack((flow0, flow1, flow2)).astype(np.float32)
        print(np.shape(img))
        plt.imshow(img/255)
        plt.show()'''
        
        
    
        
        
        
        dIx0 = np.zeros_like(frame1[:,:,0]).astype(np.float32)
        dIy0 = np.zeros_like(frame1[:,:,0]).astype(np.float32)
        
        
        dIx1 = np.zeros_like(frame1[:,:,0]).astype(np.float32)
        dIy1 = np.zeros_like(frame1[:,:,0]).astype(np.float32)
        
        dIx2 = np.zeros_like(frame1[:,:,0]).astype(np.float32)
        dIy2 = np.zeros_like(frame1[:,:,0]).astype(np.float32)
        
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)
        s = np.shape(dIx0)
        
        
        result = np.zeros_like(frame1).astype(np.float32)
        
        #for i in range(s[0]):
            #for j in range(s[1]):
                #posY = flow0[]
        
        # computes derivatives 
        for i in range(s[0]):
            for j in range(s[1]):
                if not (j==0 or j==s[1]-1 or i==0 or i==s[0]-1):
                    dIx0[i][j] = (frame1[i,j+1,0] - frame1[i,j-1,0])/2.0
                    dIx1[i][j] = (frame1[i,j+1,1] - frame1[i,j-1,1])/2.0
                    dIx2[i][j] = (frame1[i,j+1,2] - frame1[i,j-1,2])/2.0
                    
                    dIy0[i][j] = (frame1[i+1,j,0] - frame1[i-1,j,0])/2.0
                    dIy1[i][j] = (frame1[i+1,j,1] - frame1[i-1,j,1])/2.0
                    dIy2[i][j] = (frame1[i+1,j,2] - frame1[i-1,j,2])/2.0
        #print(np.shape(dIx0))
        #print(np.shape(flow0[..., 0]))
        
        # Computes dI/dt for each channe;
        #print(np.shape(frame1[..., 0]))
        
        ch0 = frame1[..., 0]-(dIx0*flow0[..., 0] + dIy0*flow0[..., 1])
        ch1 = frame1[..., 1]-(dIx1*flow1[..., 0] + dIy1*flow1[..., 1])
        ch2 = frame1[..., 2]-(dIx2*flow2[..., 0] + dIy2*flow2[..., 1])
        
        # Builds image prediction
        pred = np.dstack((ch0, ch1, ch2))
 
        pred[pred>255] = 255.0
        pred[pred<0] = 0.0
        
        imsave(pred, os.path.join(saveDir, 'test.png'))
        
        #print('Prediction:')
        pred = sp.misc.imresize(pred, 100,  'bilinear')
        #plt.imshow(pred/255)
        #plt.show()
        
        #print('Answer:')
        #plt.imshow(frame2/255)
        #plt.show()
        #plt.close()
        
        print('Error: ' + str(np.linalg.norm(pred-frame2)))
        imsave(pred-frame2, os.path.join(saveDir, 'diff.png'))

if __name__=='__main__':
    img1Path = 'C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\New_out\\360p\\video (20)_3\\719.png'
    img2Path = 'C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\New_out\\360p\\video (20)_3\\720.png'
    savePath = 'C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Tools'
    a = cv2.imread(img1Path)
    b = cv2.imread(img2Path)
    denseFlowMc(a,b,savePath)
    
  
        
        
                
                
            
    
    
