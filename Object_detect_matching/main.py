# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:47:32 2017

@author: Administrator
"""

import cv2
import wx
from gui import BaseLayout
from feature_matching1 import FeatureMatching

def main():
    capture=cv2.VideoCapture('test.mp4')
    if not(capture.isOpened()):
        capture.open()
    
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,640)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,480)
    
    app=wx.App()
    
    layout=FeatureMatchingLayout(capture=capture,title='Feature Matching',fps=30)
    layout.Show()
    app.MainLoop()

class FeatureMatchingLayout(BaseLayout):
    def _create_custom_layout(self):
        pass
    def _init_custom_layout(self):
        pass
    def _process_frame(self,frame):
        self.matching=FeatureMatching(train_image='test1.png')
        success,new_frame=self.matching.match(frame)
        if success:
            return new_frame
        else:
            return frame
        #return frame

main()    
    