# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 22:07:17 2016

@author: Administrator
"""

import numpy as np
import wx
import cv2
from gui1 import BaseLayout
from filters import PencilSketch,WarmingFilter,CoolingFilter,cartoonizer

def main(read_frame):
    capture=cv2.VideoCapture(read_frame)
    if not (capture.isOpened):
        capture.open()
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,640)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,480)
    
    app=wx.App()
    layout=FilterLayout(title='Fun with Filters',capture=capture)
    layout.Show(True)
    app.MainLoop()
    
class FilterLayout(BaseLayout):
    
    def _init_custom_layout(self):
        self.pencil_sketch=PencilSketch()
        self.warm_filter=WarmingFilter()
        self.cool_filter=CoolingFilter()
        self.cartoonizer=cartoonizer()
        
    def _create_custom_layout(self):
        pnl=wx.Panel(self,-1)
        self.mode_warm=wx.RadioButton(pnl,-1,'Warming Filter',(10,10),style=wx.RB_GROUP)
        self.mode_cool=wx.RadioButton(pnl,-1,'Cooling Filter',(10,10))
        self.mode_sketch=wx.RadioButton(pnl,-1,'Pencil Sketch',(10,10))
        self.mode_cartoon=wx.RadioButton(pnl,-1,'Cartoon',(10,10))
        hbox=wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.mode_warm,1)
        hbox.Add(self.mode_cool,1)
        hbox.Add(self.mode_sketch,1)
        hbox.Add(self.mode_cartoon,1)
        pnl.SetSizer(hbox)
        self.panels_vertical.Add(pnl,flag=wx.EXPAND | wx.BOTTOM | wx.TOP , border=1)
    
    def _process_frame(self,frame_rgb):
        if self.mode_warm.GetValue():
            frame=self.warm_filter.render(frame_rgb)
        if self.mode_cool.GetValue():
            frame=self.cool_filter.render(frame_rgb)
        if self.mode_sketch.GetValue():
            frame=self.pencil_sketch.render(frame_rgb)
        if self.mode_cartoon.GetValue():
            frame=self.cartoonizer.render(frame_rgb,0,7)
        return frame
if __name__=='__main__':
    main('test.mp4')
         
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    