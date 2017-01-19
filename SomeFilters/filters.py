# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 20:21:55 2016

@author: Administrator
"""

from scipy.interpolate import UnivariateSpline
import numpy as np
import cv2

class WarmOrCoolFilter(object):
    def __init__(self):
        #增加pixel值
        self.incr_ch_lut=self._create_LUT_8UC1([0,64,128,192,256],[0,70,140,210,256])
        #减小pixel值
        self.decr_ch_lut=self._create_LUT_8UC1([0,64,128,192,256],[0,30,80,120,192])
        
    
    
    def _create_LUT_8UC1(self,x,y):
        '''
        Curve filter,用UnivariateSpline方法，插值得到０～２５５的对应值
        x:(N,),array-like,1D array
        y:(N,),array-like,the same length as x
        
        '''
        spl=UnivariateSpline(x,y)
        return spl(xrange(256))
    
    def Warming_render(self,img_bgr):
        
       # 1.增加红色，较小蓝色
        b,g,r=cv2.split(img_bgr)
        r=cv2.LUT(r,self.incr_ch_lut).astype(np.uint8)#look up table
        b=cv2.LUT(b,self.decr_ch_lut).astype(np.uint8)
        img_bgr=cv2.merge((b,g,r))
    #2.转化为HSV，增加saturation
        h,s,v=cv2.split(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV))
        s=cv2.LUT(s,self.incr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2BGR)
    def Cooling_render(self,img_bgr):
        #与Cooling_render相反
        # 1.减小红色，增加蓝色
        b,g,r=cv2.split(img_bgr)
        r=cv2.LUT(r,self.decr_ch_lut).astype(np.uint8)#look up table
        b=cv2.LUT(b,self.incr_ch_lut).astype(np.uint8)
        img_bgr=cv2.merge((b,g,r))
    #2.转化为HSV，减小saturation
        h,s,v=cv2.split(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV))
        s=cv2.LUT(s,self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((h,s,v)),cv2.COLOR_HSV2BGR)
        
class WarmingFilter(object):
    def __init__(self):
        #增加pixel值
        self.incr_ch_lut=self._create_LUT_8UC1([0,64,128,192,256],[0,70,140,210,256])
        #减小pixel值
        self.decr_ch_lut=self._create_LUT_8UC1([0,64,128,192,256],[0,30,80,120,192])
        
    def _create_LUT_8UC1(self,x,y):
        '''
        Curve filter,用UnivariateSpline方法，插值得到０～２５５的对应值
        x:(N,),array-like,1D array
        y:(N,),array-like,the same length as x
        
        '''
        spl=UnivariateSpline(x,y)
        return spl(xrange(256))
    
    def render(self, img_rgb):
        # warming filter: increase red, decrease blue
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

class CoolingFilter(object):
    def __init__(self):
        """Initialize look-up table for curve filter"""
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self, img_rgb):
        # cooling filter: increase blue, decrease red
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        """Creates a look-up table using scipy's spline interpolation"""
        spl = UnivariateSpline(x, y)
        return spl(xrange(256))
        
class PencilSketch(object):
            
    def dodgeV2(self,image,mask):
        return cv2.divide(image,255-mask,scale=256)
        
    def renderV2(self,img_rgb):
        img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
        img_gray_inv=255-img_gray
        img_blur=cv2.GaussianBlur(img_gray_inv,(21,21),0,0)
        image_blend=self.dodgeV2(img_gray,img_blur)
        return image_blend
    
    def render(self,img_rgb):
        '''
        将图片转换为素描
        1.转化为Gray
        2.用高斯滤镜
        3.color dodge·（颜色减淡）
        '''
        img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
        img_blur=cv2.GaussianBlur(img_gray,(21,21),0,0)
        img_blend=cv2.divide(img_gray,img_blur,scale=256)
        return cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)

class cartoonizer(object):
    
    def render(self,img,numDownSample=2,numBilateralFilters=7):
        img_color=img
        #缩小图片
        for _ in xrange(numDownSample):
            img_color=cv2.pyrDown(img_color)
        #多次使用双边滤波（小数值）
        for _ in xrange(numBilateralFilters):
            img_color=cv2.bilateralFilter(img_color,9,sigmaColor=9,sigmaSpace=7)
        #放大图片
        for _ in xrange(numDownSample):
            img_color=cv2.pyrUp(img_color)
        #转化为灰度图，并用中值滤波
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img_blur=cv2.medianBlur(img_gray,7)
        #检测轮廓
        img_edge=cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        #图片转换到彩色，并且与原图结合
        img_edge=cv2.cvtColor(img_edge,cv2.COLOR_GRAY2RGB)
        return cv2.bitwise_and(img_color,img_edge)#按位与运算
'''
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
c=cartoonizer()
print frame.shape
while ret and cv2.waitKey(100) ==-1:
    a=c.render(frame,0,7)
    cv2.imshow('123',a)
    ret,frame=cap.read()
cv2.destroyAllWindows()
cap.release()
'''      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    