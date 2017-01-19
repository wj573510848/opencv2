# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 22:15:08 2017

@author: Administrator
"""
import cv2
import numpy as np

class FeatureMatching(object):
    def __init__(self,train_image):
        #定义图像特征提取方法，采用SURF，阈值为400.
        self.min_hessian=400
        self.SURF=cv2.SURF(self.min_hessian)
        #定义特征匹配方法，采用FLANN。
        FLANN_INDEX_KDTREE=0
        index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
        search_params=dict(checks=50)
        self.flann=cv2.FlannBasedMatcher(index_params,search_params)
        #输入需要匹配的图片。
        self.img_train=cv2.imread(train_image,cv2.CV_8UC1)
        if self.img_train is None:
            print "Could not find train image" + train_image
            raise SystemExit
        self.sh_train=self.img_train.shape[:2]
        self.kp_train,self.desc_train=self.SURF.detectAndCompute(self.img_train,None)

      
        self.last_hinv=np.zeros((3,3))
        self.num_frames_no_success=0
        self.max_frames_no_success=5
        self.max_error_hinv=50
    
    def match(self,frame):
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        kp_query,desc_query=self._extract_features(frame)
        good_matches=self._match_features(desc_query)
        out=self.draw_good_matches(self.img_train,frame,self.kp_train,kp_query,good_matches)
        query_corners=self._detect_corner_points(kp_query,good_matches)
        for i in xrange(len(query_corners)):
            cv2.line(out,query_corners[i],query_corners[(i+1)%4],(0,255,0),3)
        return 1,out
    
    def _extract_features(self,img):
        return self.SURF.detectAndCompute(img,None)
    
    def _match_features(self,desc_img):
        matches=self.flann.knnMatch(self.desc_train,desc_img,k=2)
        if matches is None:
            return None
        good_matches=filter(lambda x:x[0].distance<0.7*x[1].distance,matches)
        good_matches=[good_matches[i][0] for i in xrange(len(good_matches))]
        return good_matches
    
    def _detect_corner_points(self,kp_query,good_matches):
        train_points=[self.kp_train[good_matches[i].queryIdx].pt for i in xrange(len(good_matches))]
        query_points=[kp_query[good_matches[i].trainIdx].pt for i in xrange(len(good_matches))]
        H,_=cv2.findHomography(np.array(train_points),np.array(query_points),cv2.RANSAC)
        sh_train=self.img_train.shape[:2]
        train_corners=np.array([(0,0),(sh_train[1],0),(sh_train[1],sh_train[0]),(0,sh_train[0])],dtype=np.float32)
        query_corners=cv2.perspectiveTransform(train_corners[None,:,:],H)
        query_corners=map(tuple,query_corners[0])
        query_corners=[(np.int(query_corners[i][0]+self.sh_train[1]),np.int(query_corners[i][1])) for i in xrange(len(query_corners))]
        return query_corners
        
     
    def draw_good_matches(self,img1,img2,kp1,kp2,matches):
        row1,cols1=img1.shape[:2]
        row2,cols2=img2.shape[:2]
        col=cols1+cols2
        if row2>row1:
            row=col/cols2*row2
        else:
            row=col/cols2*row1
        out=np.zeros((row,col,3),dtype=np.uint8)
        out[:row1,:cols1,:]=np.dstack([img1,img1,img1])
        out[:row2,cols1:cols1+cols2,:]=np.dstack([img2,img2,img2])

        if matches is not None:
            for m in matches:
                c1,r1=kp1[m.queryIdx].pt
                c2,r2=kp2[m.trainIdx].pt
                cv2.circle(out,(int(c1),int(r1)),4,(255,0,0),1)
                cv2.circle(out,(int(c2)+cols1,int(r2)),4,(255,0,0),1)
                cv2.line(out,(int(c1),int(r1)),(int(c2)+cols1,int(r2)),(255,0,0),1)
               
        out=cv2.resize(out,(cols2,row2))
        return out
if __name__=="__main__":
    a=FeatureMatching('t1.jpg')
    frame=cv2.imread('t2.jpg')
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    a.match(frame)
    