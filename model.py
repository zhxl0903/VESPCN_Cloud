import tensorflow as tf
import numpy as np
import math
import time
import os
import glob
import cv2
import scipy as sp

from utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    checkimage,
    imsave,
    imread,
    load_data,
    preprocess,
    modcrop
)
from PSNR import psnr

class ESPCN(object):
    
    '''
       This method is contructor for class ESPSCN.
       Given: 
            sess: Session object for this model
            image_size: size of image for training
            is_train: True iff training
            train_mode: 0 is spatial transformer only
                        1 is single frame VESPSCN with Early Fusion No MC
                        2 is Early Fusion VESPCN with MC
                        3 is Bicubic (No Training Required)
                        4 is SRCNN
            scale: upscaling ratio for super resolution
            batch_size: batch size for training
            c_dim: number of channels of each input image
            config: config object
       Returns: None
    '''                   
    def __init__(self,
                 image_size,
                 is_train,
                 train_mode,
                 scale,
                 batch_size,
                 c_dim,
                 load_existing_data,
                 device,
                 learn_rate,
                 data_list
                 ):
        
        # Initializes layer memory dictionary
        self.layerOutputs = dict()
        self.load_existing_data = load_existing_data
        self.device = device
        self.learn_rate = learn_rate
        self.data_list = data_list
 
        self.image_size = image_size
        self.is_train = is_train
        self.c_dim = c_dim
        self.scale = scale
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.build_model()
        
    '''
       This method builds network based on training mode. Placeholders are
       setup based on training mode and is_train. Predictions are obtained
       by calling method model. Saver is initialized and loss functions are
       initialized based on training mode.
       
       Given: None
       Returns: None
    '''
    def build_model(self):

        with tf.device(self.device):
            if self.is_train:

                if self.train_mode == 0:
                    self.images_curr_prev = tf.reshape(self.data_list[0],
                                                       [self.batch_size, self.image_size,
                                                        self.image_size, 2*self.c_dim])
                    self.labels = tf.reshape(self.data_list[1],
                                             [self.batch_size, self.image_size*self.scale,
                                              self.image_size*self.scale, self.c_dim])
                elif self.train_mode == 1 or self.train_mode == 6 or self.train_mode == 4:
                    self.images_in = tf.reshape(self.data_list[0],
                                                [self.batch_size, self.image_size, self.image_size, self.c_dim])
                    self.labels = tf.reshape(self.data_list[1],
                                             [self.batch_size, self.image_size*self.scale,
                                              self.image_size*self.scale, self.c_dim])
                else:
                    self.images_prev_curr = tf.reshape(self.data_list[0],
                                                       [self.batch_size, self.image_size, self.image_size,
                                                        2*self.c_dim])
                    self.images_next_curr = tf.reshape(self.data_list[1],
                                                       [self.batch_size, self.image_size,
                                                        self.image_size, 2*self.c_dim])
                    self.labels = tf.reshape(self.data_list[2],
                                             [self.batch_size, self.image_size*self.scale,
                                              self.image_size*self.scale, self.c_dim])
            else:
                # Computes shape of placeholder for image feed from sample image
                # loaded
                print('Train Mode:', self.train_mode)
                data = load_data(self.is_train, self.train_mode)
                input_ = imread(data[0][0])
                self.h, self.w, c = input_.shape

                if self.train_mode == 0:
                    self.images_curr_prev = self.data_list[0]
                    self.labels = self.data_list[1]
                elif self.train_mode == 1 or self.train_mode == 3 or self.train_mode == 6 or self.train_mode == 4:
                    self.images_in = self.data_list[0]
                    self.labels = self.data_list[1]
                else:
                    self.images_prev_curr = self.data_list[0]
                    self.images_next_curr = self.data_list[1]
                    self.labels = self.data_list[2]

            if self.train_mode == 0 or self.train_mode == 1 or \
                self.train_mode == 3 or self.train_mode == 4 or \
                self.train_mode == 6:
                with tf.name_scope('model outputs'):
                    self.pred = self.model()
            else:
                with tf.name_scope('model outputs'):
                    self.pred, self.imgPrev, self.imgNext = self.model()

            # Prepares loss function based on training mode
            if self.train_mode == 0:

                # Defines loss function for training a single spatial transformer
                with tf.name_scope('loss function'):
                    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
            elif self.train_mode == 1 or self.train_mode == 4 or self.train_mode == 6:

                # Defines loss function for training subpixel convnet
                with tf.name_scope('loss function'):
                    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
                print('Mode 1/4/6: Mean-Squared Loss Activated')
            elif self.train_mode == 2 or self.train_mode == 5:

                # Defines loss function for training in unison
                with tf.name_scope('loss function'):
                    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred)) + \
                                               0.01*tf.reduce_mean(tf.square(self.imgPrev-self.images_prev_curr[:, :, :, 0:self.c_dim])) + \
                                               0.01*tf.reduce_mean(tf.square(self.imgNext - self.images_prev_curr[:, :, :, 0:self.c_dim]))

            # Generates summary scalar for cost
            tf.summary.scalar("cost", self.loss)
        if self.train_mode != 3:

            with tf.name_scope('global_step'):
                global_step = tf.train.get_or_create_global_step()
                self.global_step = global_step

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss,
                                                                                               global_step=global_step)
        # Merges all summaries
        self.summary_op = tf.summary.merge_all()
        
    '''
    This method constructs spatial transformer given frameSet.
    
    Inputs:
    frameSet: frameSet containing 2 images: current frame and neighbouring frame
              Tensor of dimension: [nBatch, imgH, imgW, 6]
              
    reuse: True iff Weights are the same as network where reuse is false
    
    Returns: Output image of this spatial transformer network
             Tensor of dimension: [nBatch, imgH, imgW, 3]
    '''
    def spatial_transformer(self, frameSet, reuse=False):
        
        # Zero initialization
        biasInitializer = tf.zeros_initializer()
        
        # Orthogonal initialization with gain sqrt(2)
        weight_init = tf.orthogonal_initializer(np.sqrt(2))
        
        # Course flow
        t1_course_l1 = tf.layers.conv2d(frameSet,  24, 5, padding='same',
                                        strides=(2, 2),
                                        activation=tf.nn.relu,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l1', reuse=reuse)

        t1_course_l2 = tf.layers.conv2d(t1_course_l1,  24, 3, padding='same',
                                        strides=(1, 1),
                                        activation=tf.nn.relu,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l2', reuse=reuse)
        t1_course_l3 = tf.layers.conv2d(t1_course_l2,  24, 5, padding='same',
                                        strides=(2, 2),
                                        activation=tf.nn.relu,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l3', reuse=reuse)
        t1_course_l4 = tf.layers.conv2d(t1_course_l3,  24, 3, padding='same',
                                        strides=(1, 1),
                                        activation=tf.nn.relu,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l4', reuse=reuse)
        t1_course_l5 = tf.layers.conv2d(t1_course_l4,  32, 3, padding='same',
                                        strides=(1, 1),
                                        activation=tf.nn.tanh,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l5', reuse=reuse)
        
        # Defines Course Flow Output
        # Output shape: (-1, l, w, 2)
        t1_course_out = self.PS2(t1_course_l5, 4, 2)
        
        if not reuse:
            self.layerOutputs['courseFlow'] = t1_course_out
        
        # Course Warping
        # Gets target image to be warped
        targetImg = frameSet[:, :, :, self.c_dim:self.c_dim*2]
        
        # Generates tensor of dimension [-1, h, w, 3+2]
        t1_course_warp_in = tf.concat([targetImg, t1_course_out], 3)
        
        # Applies warping using 3D convolution to estimate image at time
        # t=t
        # Kernel size 3 is used to apply flow to image based on neighbouring
        # flows of pixel
        t1_course_warp = tf.layers.conv2d(t1_course_warp_in, 3, 3,
                                          padding='same',
                                          activation=tf.nn.tanh,
                                          kernel_initializer=weight_init,
                                          bias_initializer=biasInitializer,
                                          name='t1_course_warp', reuse=reuse)
        
        # Fine flow 
        # Combines images input, course flow estimation, 
        # and course image estimation along dimension 3
        t1_fine_in = tf.concat([frameSet, t1_course_warp,
                                t1_course_out], 3)
        
        t1_fine_l1 = tf.layers.conv2d(t1_fine_in,  24, 5, padding='same',
                                      strides=(2,2),
                                      activation=tf.nn.relu,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l1', reuse=reuse)
        
        t1_fine_l2 = tf.layers.conv2d(t1_fine_l1,  24, 3, padding='same',
                                      strides=(1, 1),
                                      activation=tf.nn.relu,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l2', reuse=reuse)
        
        t1_fine_l3 = tf.layers.conv2d(t1_fine_l2,  24, 3, padding='same',
                                      strides=(1, 1),
                                      activation=tf.nn.relu,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l3', reuse=reuse)
        
        t1_fine_l4 = tf.layers.conv2d(t1_fine_l3,  24, 3, padding='same',
                                      strides=(1, 1),
                                      activation=tf.nn.relu,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l4', reuse=reuse)
        
        t1_fine_l5 = tf.layers.conv2d(t1_fine_l4,  8, 3, padding='same',
                                      strides=(1, 1),
                                      activation=tf.nn.tanh,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l5', reuse=reuse)
        
        # Output shape(-1, l, w, 2)
        t1_fine_out = self.PS2(t1_fine_l5, 2, 2)
        
        if not reuse:
            self.layerOutputs['fineFlow'] = t1_fine_out
        
        # Combines fine flow and course flow estimates
        # Output shape(-1, l, w, 2)
        t1_combined_flow = t1_course_out + t1_fine_out
        
        # Fine Warping
        # Concatnates target image and combined flow channels
        t1_fine_warp_in = tf.concat([targetImg, t1_combined_flow], 3)
        
        # Applies warping using 2D convolution layer to estimate image at time
        # t=t
        # Kernel size 3 is used to apply flow based on neighbouring flows of
        # pixel
        # Output shape: (batchSize, h, w, c_dim)
        t1_fine_warp = tf.layers.conv2d(t1_fine_warp_in, 3, 3, padding='same',
                                        activation=tf.nn.tanh, 
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_fine_warp', reuse=reuse)
        
        # Resizes using billinear interpolation
        # Output shape: (batchSize, h, w, c_dim)
        return t1_fine_warp
        
    '''
    This method generates a network model given self.train_mode
    train mode = 0: model for 1 spatial transformer
    train mode = 1: model for single frame ESPCN 
    train mode = 2: model for VESPCN with 2 spatial transformers taking 2 
                    images each
    
    Returns: Output of network if train mode is 0 or 1
             Output of network and output of 2 spatial transformers of network
             if train mode is 2
    '''
    def model(self):

        # Generates motion compensated images from previous and next images
        # using 2 spatial transformers
           
        # Initializes spatial transformer if training mode is 0 or 2
        if self.train_mode == 2 or self.train_mode == 5:

            with tf.name_scope('Previous-Image MC'):
                imgPrev = self.spatial_transformer(self.images_prev_curr,
                                                   reuse=False)
            with tf.name_scope('Next-Image MC'):
                imgNext = self.spatial_transformer(self.images_next_curr,
                                                   reuse=True)
            with tf.name_scope('Target-Image'):
                targetImg = self.images_prev_curr[:, :, :, 0:self.c_dim]
            with tf.name_scope('Stacked Input'):
                imgSet = tf.concat([imgPrev, targetImg, imgNext], 3)
        elif self.train_mode == 0:

            with tf.name_scope('Spatial Transformer'):
                motionCompensatedImgOut = self.spatial_transformer(self.
                                                                   images_curr_prev,
                                                                   reuse=False)
        else:
            with tf.name_scope('Image Input'):
                imgSet = self.images_in

        # wInitializer1 = tf.random_normal_initializer(stddev=np.sqrt(2.0/25/3))
        # wInitializer2 = tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64))
        # wInitializer3 = tf.random_normal_initializer(stddev=np.sqrt(2.0/9/32))
        wInitializer1 = tf.orthogonal_initializer(np.sqrt(2))
        wInitializer2 = tf.orthogonal_initializer(np.sqrt(2))
        wInitializer3 = tf.orthogonal_initializer(np.sqrt(2))
       
        biasInitializer = tf.zeros_initializer()
       
        if self.train_mode == 2 or self.train_mode == 5:
           
            # Connects early fusion network with spatial transformer
            # and subpixel convnet. For collapsing to temporal depth of 1,
            # number of channels produced is 24 by VESPCN paper

            with tf.name_scope('Early Fusion Layer'):
                EarlyFusion = tf.layers.conv2d(imgSet,  24, 3, padding='same',
                                               activation=tf.nn.relu,
                                               kernel_initializer=wInitializer1,
                                               bias_initializer=biasInitializer,
                                               name='EF1')

                subPixelIn = EarlyFusion
       
        elif self.train_mode == 1 or self.train_mode == 6:

            with tf.name_scope('ESPCN in'):
                # Connects subpixel convnet to placeholder for feeding single images
                subPixelIn = tf.layers.conv2d(imgSet,  24, 3, padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=wInitializer1,
                                              bias_initializer=biasInitializer,
                                              name='subPixelIn')
        elif self.train_mode == 3:
           
            # Sets height and width for resizing based on is_train
            if self.is_train:
               
                # Sets to training data patch size
                height = self.image_size
                width = self.image_size
            else:
               
                # Sets to image size
                height = self.h
                width = self.w
            with tf.name_scope('Bicubic Upscaling'):
                biCubic = tf.image.resize_images(imgSet, [height*self.scale,
                                                 width*self.scale],
                                                 method=tf.image.ResizeMethod.BICUBIC)
    
        # TO DO: Enable all layers in every step but
        # adjust inputs and outputs to layers depending on
        # mode so the entire model is accessible through checkpoint
        # Builds subpixel net if train mode is 1 or 2
        if self.train_mode == 1 or self.train_mode == 2 or self.train_mode == 5 \
           or self.train_mode == 6:
            with tf.name_scope('ESPCN'):
                conv1 = tf.layers.conv2d(subPixelIn,  24, 3, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer1,
                                         bias_initializer=biasInitializer,
                                         name='subPixelL1')
                conv2 = tf.layers.conv2d(conv1,  24, 3, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer1,
                                         bias_initializer=biasInitializer,
                                         name='subPixelL2')
                conv3 = tf.layers.conv2d(conv2,  24, 3, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer1,
                                         bias_initializer=biasInitializer,
                                         name='subPixelL3')
                conv4 = tf.layers.conv2d(conv3,  24, 3, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer1,
                                         bias_initializer=biasInitializer,
                                         name='subPixelL4')
                conv5 = tf.layers.conv2d(conv4,  24, 3, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer2,
                                         bias_initializer=biasInitializer,
                                         name='subPixelL5')
                conv6 = tf.layers.conv2d(conv5,  24, 3, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer2,
                                         bias_initializer=biasInitializer,
                                         name='subPixelL6')
                conv7 = tf.layers.conv2d(conv6,  24, 3, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer2,
                                         bias_initializer=biasInitializer,
                                         name='subPixelL7')

                conv8 = tf.layers.conv2d(conv7,
                                         self.c_dim * self.scale * self.scale,
                                         3, padding='same', activation=None,
                                         kernel_initializer=wInitializer3,
                                         bias_initializer=biasInitializer,
                                         name='subPixelL8')

                ps = self.PS(conv8, self.scale)
        elif self.train_mode == 4:

            with tf.name_scope('SRCNN'):
                # Builds SRCNN network if train_mode is 4
                conv1 = tf.layers.conv2d(imgSet,  64, 9, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer1,
                                         bias_initializer=biasInitializer,
                                         name='SRCNN1')
                conv2 = tf.layers.conv2d(conv1,  32, 1, padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=wInitializer1,
                                         bias_initializer=biasInitializer,
                                         name='SRCNN2')
                conv3 = tf.layers.conv2d(conv2,  self.c_dim, 5, padding='same',
                                         activation=None,
                                         kernel_initializer=wInitializer1,
                                         bias_initializer=biasInitializer,
                                         name='SRCNN3')

        # Returns network output given self.train_mode

        with tf.name_scope('Output(s)'):
            if self.train_mode == 0:
                return motionCompensatedImgOut
            elif self.train_mode == 1 or self.train_mode == 6:
                return tf.nn.tanh(ps)
            elif self.train_mode == 3:
                return (biCubic)
            elif self.train_mode == 4:
                return conv3
            else:
                return tf.nn.tanh(ps), imgPrev, imgNext

    '''
    This method serves as a helper method for PS ad PS2 in the case of 
    processing training images. Input tensor X in PS and PS2 has r*r*cdim 
    channels which is split into cdim tensors each with r*r channels which
    are processed one at a time by calling this method. 
    
    Note: batch size can be more than 1 in this method
    
    Input:
        I: Tensor of dimension (batch_size, a, b, r*r)
        r: upscaling ratio
    Returns:
        Tensor of dimension (batch_size, a*r, b*r, 1) 
    '''
    # NOTE: train with with batch size
    def _phase_shift(self, I, r):
        
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (-1, a, b, r, r))
        
        # a, [bsize, b, r, r]
        X = tf.split(X, a, 1)  
        
        # bsize, b, a*r, r
        X = tf.concat([tf.squeeze(x) for x in X], 2)  
        
        # b, [bsize, a*r, r]
        X = tf.split(X, b, 1)  
        
        # bsize, a*r, b*r
        X = tf.concat([tf.squeeze(x) for x in X], 2)  
        return tf.reshape(X, (-1, a*r, b*r, 1))

    '''
    This method serves as a helper method for PS ad PS2 in the case of 
    processing test images. Input tensor X in PS and PS2 has r*r*cdim 
    channels which is split into cdim tensors each with r*r channels which
    are processed one at a time by calling this method. 
    
    Note: only single image batches are supported during testing
    
    Input:
        I: Tensor of dimension (1, a, b, r*r)
        r: upscaling ratio
    Returns:
        Tensor of dimesnion (1, a*r, b*r, 1) 
    '''
    # NOTE:test without batchsize
    def _phase_shift_test(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        
        # a, [bsize, b, r, r]
        X = tf.split(X, a, 1)  
        
        # bsize, b, a*r, r
        X = tf.concat([tf.squeeze(x) for x in X], 1)  
        
        # b, [bsize, a*r, r]
        X = tf.split(X, b, 0)  
        
        # bsize, a*r, b*r
        X = tf.concat([tf.squeeze(x) for x in X], 1)  
        return tf.reshape(X, (1, a*r, b*r, 1))
        
    '''
       Performs phase shift operation for tensor of dimension
       (batch_size, img_height, img_width, 3*r*r)
       
       Inputs:
       X: tensor of dimension (batch_size, img_height, img_width, 3*r*r)
       r: upscaling factor
       c_dim: c_dim of X
       
       Returns:
       Tensor of shape (batchs_size, img_height*r, img_width*r, 3)
    '''
    def PS(self, X, r):
        
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            
            # Does concat RGB
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) 
        else:
            
            # Does concat RGB
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) 
        return X
    
    '''
       Performs phase shift operation for tensor of dimension
       (batch_size, img_height, img_width, c_dim)
       
       Inputs:
       X: tensor of dimension (batch_size, img_height, img_width, c_dim*r*r)
       r: upscaling factor
       c_dim: c_dim of X
       
       Returns:
       Tensor of shape (batchs_size, img_height*r, img_width*r, c_dim)
    '''
    def PS2(self, X, r, c_dim):
        
        # Main OP that you can arbitrarily use in you tensorflow code
        
        # Evenly splits Xc into c_dim parts along axis 3 (# of channels)
        Xc = tf.split(X, c_dim, 3)
        if self.is_train:
            
            # Does the concat RGB
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) 
        else:
            # Does the concat RGB
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3)
        return X
    
    '''
       This method performs training / testing operations given config.
       Training / Testing is supported for all 3 modes from self.train_mode.
       
       See class __init__() for additional a description of training modes.
       
       Input:
            config: config object of this class
       
    '''
    def train(self, config, sess):
        
        # NOTE : if train, the nx, ny are ingnored
        
        #  Sets optimizer to be Adam optimizer
        
        # self.sess.run(tf.global_variables_initializer())

        # counter = 0
        # time_ = time.time()
        
        """if self.train_mode != 3:
            self.load(config.checkpoint_dir)"""
        
        # Train
        if config.is_train:

            variables = [self.global_step,  self.train_op]
            step, _ = sess.run(variables)

            if step > 0 and step % 10 == 0:
                loss = sess.run(self.loss)
                print("step: " + str(step) + " loss: " + str(loss))

        # Test
        else:
            print("Now Start Testing...")
            
            psnrLst = []
            errorLst = []
            
            if self.train_mode == 0:
                
                # Obtains course flow tensor
                courseFlow = self.layerOutputs['courseFlow']
            
                # Obtains fine flow tensor
                fineFlow = self.layerOutputs['fineFlow']
                
                # Performs testing for mode 0
                # Saves each motion compensated image generated from 
                # from a frame set to result_dir
                # Note: Each testing image must have the same size
                for i in range(len(input_)):
                    result, courseF, fineF = sess.run([self.pred,
                                                       courseFlow, fineFlow],
                                                      feed_dict=
                                                      {self.images_curr_prev:
                                                       input_[i].reshape(1,
                                                                         self.h, self.w, 2*self.c_dim)})

                    ''' result = self.pred.eval({self.images_curr_prev: 
                                              input_[i].reshape(1,
                                              self.h, self.w, 2*self.c_dim)})'''
    
                    original = input_[i].reshape(1, self.h, self.w,
                                                 2*self.c_dim)
                    original = original[0, :, :, 0:self.c_dim]
                    
                    errorMap = 1 - (original - result)
                    
                    # Computes mean error and appends to errorLst for 
                    # mean error computation
                    error = np.mean(np.square(original-result), axis = None)
                    errorLst.append(error)
                    print('Error on frame set ', i, ' : ', error)
                    
                    x = np.squeeze(result)
                    errorMapOut = np.squeeze(errorMap)
                    
                    courseFOut = np.squeeze(courseF)
                    fineFOut = np.squeeze(fineF)
                    
                    # Computes norm of course flow
                    courseNorm = np.sqrt(np.square(courseFOut[:, :, 0]) +
                                 np.square(courseFOut[:, :, 1]))
                    
                    # Computes norm of fine flow
                    fineNorm = np.sqrt(np.square(fineFOut[:, :, 0]) +
                                 np.square(fineFOut[:, :, 1]))
                    
                    
                    # Inverts norm maps for better view
                    courseNorm = 1 - courseNorm
                    fineNorm = 1 - fineNorm
                    
                    # Inverts course and fine flow maps
                    courseFOut = 1 - courseFOut
                    fineFOut = 1 - fineFOut
                                        
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i) +
                           '.png', config)
                    imsave(errorMapOut, config.result_dir +
                           '/result_errorMap' + str(i) + '.png', config)
                    imsave(courseFOut[:, :, 0], config.result_dir +
                           '/result_courseMap0_' + str(i) + '.png', config)
                    imsave(courseFOut[:, :, 1], config.result_dir +
                           '/result_courseMap1_' + str(i) + '.png', config)
                    imsave(fineFOut[:, :, 0], config.result_dir +
                           '/result_fineMap0_' + str(i) + '.png', config)
                    imsave(fineFOut[:, :, 1], config.result_dir +
                           '/result_fineMap1_' + str(i) + '.png', config)
                    imsave(courseNorm, config.result_dir +
                           '/result_courseNorm_' + str(i) + '.png', config)
                    imsave(fineNorm, config.result_dir +
                           '/result_fineNorm_' + str(i) + '.png', config)
                print('Mean Avg Error: ', np.mean(errorLst))
            elif self.train_mode == 1:
                
                # Performs testing for mode 1
                # Saves hr images for each input image to resut_dir
                for i in range(len(input_)):
                    result = self.pred.eval({self.images_in:
                                             input_[i].reshape(1,
                                                               self.h, self.w, self.c_dim)})
                    
                    # Obtains original input image from input_
                    #orgInput = input_[i].reshape(self.h,
                                        #self.w, self.c_dim)
                    
                    # Denormalizes input image
                    #orgInput = orgInput * 255.0
                    
                    # Removes batch size axis from tensor
                    x = np.squeeze(result)

                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i)+'.png',
                           config)
                    
                    # Loads output image
                    #data_LR = glob.glob('./result/result'+str(i)+'.png')
                    #lr = cv2.imread(data_LR[0])
                    
                    #print('Super resolution shape: ', np.shape(x))
                    #print('Original input shape: ', np.shape(orgInput))
                    
                    # Computes low resolution image from super resolution image
                    # by downscaling by self.scale
                    #lr = cv2.resize(lr, None,fx = 1.0/self.scale, 
                                            #fy = 1.0/self.scale,
                                            #interpolation = cv2.INTER_CUBIC)
                    
                    # Computes and prints PSNR ratio 
                    # Appends psnr val to psnrLst for mean computation
                    #psnrVal = psnr(lr, orgInput, scale = self.scale)
                    #psnrLst.append(psnrVal)
                    #print('Image ', i, ' PSNR: ', psnrVal )
                    
                    # Prints shape of output image and saves output image
                    
                    
                #print('Average PSNR: ', np.mean(psnrLst))
            elif self.train_mode == 2:
                
                # Performs testing for mode 2
                # Saves each hr images generated from a consecutive frame set
                # to result_dir
                for i in range(len(input_)):
                    
                    # Prepares current and previous frames
                    curr_prev = input_[i, :, :, 0:2*self.c_dim].reshape(1,
                                                                        self.h, self.w, 2*self.c_dim)
                    
                    # Prepares stack of current and next frames
                    curr_next = np.concatenate((input_[i, :, :, 0:self.c_dim],
                                                input_[i, :, :,
                                                       2*self.c_dim:3*self.c_dim]),
                                               axis=2).reshape(1,
                                                               self.h, self.w, 2*self.c_dim)
    
                    # Fetches normalized original image from input_
                    # and restores pixel values to between 0-255
                    #curr_img = input_[i, :, :, 0:self.c_dim].reshape(self.h,
                                     #self.w, self.c_dim)      
                    
                    #curr_img = curr_img * 255.0
                    
                    # Generates output image
                    result = self.pred.eval({self.images_prev_curr: curr_prev,
                                             self.images_next_curr: curr_next})

                    # Removes batch size dimension from result
                    # and denormalizes result image 
                    x = np.squeeze(result)
                    
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir +
                           '/result'+str(i) + '.png',
                           config)
                    
                    # Loads output image
                    #data_LR = glob.glob('./result/result'+str(i)+'.png')
                    #lr = cv2.imread(data_LR[0])
                    
                    # Downscales output image for psnr computation
                    #lr = cv2.resize(lr, None,fx = 1.0/self.scale, 
                                            #fy = 1.0/self.scale,
                                            #interpolation = cv2.INTER_CUBIC)
                    
                    # Computes psnr and appends psnr to psnrLst for mean
                    # computation
                    #psnrVal = psnr(lr, curr_img, scale = self.scale)
                    #psnrLst.append(psnrVal)
                    #print('Image ', i, ' PSNR: ', psnrVal )
                    
                    #print(lr)
                    #print(curr_img)
                #print('Average PSNR: ', np.mean(psnrLst))
            elif self.train_mode == 3:
                
                # Performs testing for mode 1
                # Saves hr images for each input image to resut_dir
                for i in range(len(input_)):
                    result = self.pred.eval({self.images_in:
                                             input_[i].reshape(1,
                                                               self.h, self.w, self.c_dim)})
    
                    # Obtains original input image from input_
                    orgInput = input_[i].reshape(self.h,
                                                 self.w, self.c_dim)
                    
                    # Denormalizes input image
                    orgInput = orgInput * 255.0
                    
                    x = np.squeeze(result)
                    
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i)+'.png',
                           config)
                    
                    # Loads output image
                    data_LR = glob.glob('./result/result'+str(i)+'.png')
                    lr = cv2.imread(data_LR[0])
                    
                    print('Super resolution shape: ', np.shape(x))
                    print('Original input shape: ', np.shape(orgInput))
                    
                    # Computes low resolution image from super resolution image
                    # by downscaling by self.scale
                    lr = cv2.resize(lr, None, fx=1.0/self.scale,
                                    fy=1.0/self.scale,
                                    interpolation=cv2.INTER_CUBIC)
                    
                    # Computes and prints PSNR ratio 
                    # Appends psnr val to psnrLst for mean computation
                    psnrVal = psnr(lr, orgInput, scale=self.scale)
                    psnrLst.append(psnrVal)
                    print('Image ', i, ' PSNR: ', psnrVal)
                    
                    # Prints shape of output image and saves output image
                    
                print('Average PSNR: ', np.mean(psnrLst))
            elif self.train_mode == 4:
                
                for i in range(len(input_)):
                    
                    # Upscales image using bicubic interpolatiobn
                    inp = input_[i].reshape(self.h, self.w, self.c_dim)
                    inp = sp.misc.imresize(inp,
                          (self.h*self.scale,
                           self.w*self.scale), interp='bicubic')
                    
                    result = self.pred.eval({self.images_in:
                                             inp.reshape(1,
                                                         self.h*self.scale, self.w*self.scale, self.c_dim)})

                    x = np.squeeze(result)
                    # Filters image values to be in [0,1]
                    #x = np.clip(np.squeeze(result), 0, 1)
                    
                    print('Saving image ', i)
                    print('Shape of output image: ', x.shape)
                    
                    imsave(x, config.result_dir+'/result'+str(i)+'.png',
                           config)
            elif self.train_mode == 5:
                # Performs testing for mode 2
                # Saves each hr images generated from a consecutive frame set
                # to result_dir
                
                count = 0
                
                for i in range(len(input_)):
                    
                    print ('Working on dataset ' + str(i) + ' ...')
                    
                    # Gets folder name
                    folderName = os.path.basename(os.path.split(paths_[count])[0])
                    
                    folder = os.path.join(config.result_dir, folderName)
                    os.makedirs(folder)
                    for j in range(len(input_[i])):

                        # Prepares current and previous frames
                        curr_prev = input_[i][j, :, :, 0:2*self.c_dim].reshape(1,
                                                 self.h, self.w, 2*self.c_dim)
                        
                        # Prepares stack of current and next frames
                        curr_next = np.concatenate((input_[i][j, :, :, 0:self.c_dim],
                                                    input_[i][j, :, :,
                                                              2*self.c_dim:3*self.c_dim]),
                                                   axis=2).reshape(1,
                                                                   self.h, self.w, 2*self.c_dim)

                        # Generates output image
                        result = self.pred.eval({self.images_prev_curr: curr_prev,
                                                 self.images_next_curr: curr_next})
    
                        # Removes batch size dimension from result
                        # and de-normalizes result image
                        x = np.squeeze(result)
                        
                        print('Shape of output image: ', x.shape)
                        imsave(x, folder + \
                               '//result'+str(j) + '.png',
                               config)
                        count = count + 1
            elif self.train_mode == 6:        
                
                count = 0
                for i in range(len(input_)):
                    
                    print ('Working on dataset ' + str(i) + ' ...')
                    
                    # Gets folder name
                    folderName = os.path.basename(os.path.split(paths_[count])[0])
                    
                    folder = os.path.join(config.result_dir, folderName)
                    os.makedirs(folder)
                    
                    for j in range(len(input_[i])):
                        result = self.pred.eval({self.images_in:
                                                 input_[i][j].reshape(1,
                                                                      self.h, self.w, self.c_dim)})

                        # Removes batch size axis from tensor
                        x = np.squeeze(result)

                        print('Shape of output image: ', x.shape)
                        imsave(x, folder + '//result'+str(j)+'.png',
                               config)
                        count = count + 1
     
                
            
    '''
    This method performs model loading given checkpoint_dir. Model
    is loaded based on self.image_size and self.scale from
    different directories. Different models are loaded based on
    self.train_mode. See class __init__() method for a description
    of different training modes.
    
    Input:
        checkpoint_dir: directory to load checkpoint
    
    '''
    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        
        print("\nReading Checkpoints.....\n\n")
        
        
        model_dir = ""
        
        # gives model name training data size and scale based on training mode
        if(self.train_mode == 0):
            model_dir = "%s_%s_%s" % ("espcn", self.image_size, self.scale)
        elif(self.train_mode == 1 or self.train_mode == 6):
            model_dir = "%s_%s_%s" % ("vespcn_subpixel_no_mc",
                                      self.image_size, self.scale)
        elif(self.train_mode == 2 or self.train_mode == 5):
            model_dir = "%s_%s_%s" % ("vespcn",
                                      self.image_size, self.scale)
        elif(self.train_mode == 4):
            model_dir = "%s_%s_%s" % ("srcnn",
                                      self.image_size, self.scale)
        
            
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Checks if checkpoints exists 
        if ckpt and ckpt.model_checkpoint_path:
            
            # converts unicode to string
            ckpt_path = str(ckpt.model_checkpoint_path)
            
            # Loads model from ckt_path
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
            if(self.train_mode == 5):
                exit
    
    '''
    This method performs model saving given checkpoint_dir. Model
    is saved based on self.image_size and self.scale into different
    directories, Different models are saved based on self.train_mode.
    See class __init__() method for a description of different training modes. 
    
    Input:
        checkpoint_dir: directory to load checkpoint
    
    '''
    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        
        model_name = ""
        model_dir = ""
        
        # gives model name by training data size and scale
        if (self.train_mode == 0):
            model_name = "ESPCN.model"
            model_dir = "%s_%s_%s" % ("espcn", self.image_size,self.scale)
        elif (self.train_mode == 1):
            model_name = "VESPCN_Subpixel_NO_MC.model"
            model_dir = "%s_%s_%s" % ("vespcn_subpixel_no_mc",
                                      self.image_size,self.scale)
        elif (self.train_mode == 2):
            model_name = "VESPCN.model"
            model_dir = "%s_%s_%s" % ("vespcn",
                                      self.image_size,self.scale)
        elif (self.train_mode == 4):
            model_name = "SRCNN.model"
            model_dir = "%s_%s_%s" % ("srcnn",
                                      self.image_size,self.scale)
            
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        # Checks if model checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)
        
        # Saves model to checkpoint_dir
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
