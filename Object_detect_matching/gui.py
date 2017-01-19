# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 21:09:30 2016

@author: Administrator
"""

from abc import ABCMeta,abstractmethod
import wx
import time
import cv2

class Metal(wx.Frame):
    pass
class BaseLayout(Metal):
    __metaclass__=ABCMeta    
    def __init__(self,capture,id=-1,fps=10,parent=None,title=None):
        self.capture=capture
        self.fps=fps
        success,frame=self.capture.read()
        if not success:
            raise SystemExit
        self.imgHeight,self.imgWidth=frame.shape[:2]
        self.bmp=wx.BitmapFromBuffer(self.imgWidth,self.imgHeight,frame)
        wx.Frame.__init__(self,parent,id,title,size=(self.imgWidth,self.imgHeight))
        self._init_base_layout()
        self._create_base_layout()
        
    def _init_base_layout(self):
        self.timer=wx.Timer(self)
        self.timer.Start(1000./self.fps)
        self.Bind(wx.EVT_TIMER,self._on_next_frame)
        self._init_custom_layout()
       
        
    def _on_next_frame(self,event):
        ret,frame=self.capture.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if ret:
            frame=self._process_frame(frame)
        self.bmp.CopyFromBuffer(frame)
        self.Refresh(eraseBackground=False)
    
    def _on_paint(self,event):
        deviceContext=wx.BufferedPaintDC(self.pnl)
        deviceContext.DrawBitmap(self.bmp,0,0)
    
    def _create_base_layout(self):
        self.pnl=wx.Panel(self,size=(self.imgWidth,self.imgHeight))
        self.pnl.SetBackgroundColour(wx.BLACK)
        self.pnl.Bind(wx.EVT_PAINT,self._on_paint)
        self.panels_vertical=wx.BoxSizer(wx.VERTICAL)
        self.panels_vertical.Add(self.pnl,1,flag=wx.EXPAND | wx.TOP, border=1)
        self._create_custom_layout()
        self.SetMinSize((self.imgWidth,self.imgHeight))
        self.SetSizer(self.panels_vertical)
        self.Centre()
        
    @abstractmethod
    def _init_custom_layout(self):
        pass
    @abstractmethod
    def _create_custom_layout(self):
        pass
    @abstractmethod
    def _process_frame(self):
        pass
    
    
    
    
        
        
        
        
        
        
    