#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plotpicture_v02.py
@Time    :   2019/04/11 14:51:37
@Author  :   skzhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, zhaogroup-lala
@Desc    :   None
'''


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import time, os, pickle
from multiprocessing import Queue
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import call.T1_fit as tf

path_str = False

class PlotPicture():

    def __init__(self,q=None,pathaddress=None):
        self.q = q
        self.pathaddress = pathaddress
    def ColorPicture(self,tag=None):
        data = np.load(''.join((self.pathaddress, '.npz')))
        #data = np.array(data)
        r = np.asarray(data['row'])
        c = np.asarray(data['col'])
        m = np.asarray(data['s'])
        
        fig = plt.gcf()
        a = plt.getp(fig,'axes')

        if tag=='db':
            m2 = 20*np.log10(np.abs(m))
            ax1 = a[0]
            extent = [np.min(c), np.max(c), np.min(r), np.max(r)] 
            im1 = ax1.imshow(m2,extent=extent,aspect='auto',origin='lower',animated=True)
            im1.set_clim(np.min(m2),np.max(m2))
            plt.colorbar(im1,ax=ax1)

            if np.angle(m).any():
                ax2 = a[1]
                extent = [np.min(c), np.max(c), np.min(r), np.max(r)] 
                im2 = ax2.imshow(np.angle(m),cmap='hsv',vmin=-np.pi,vmax=np.pi,extent=extent,aspect='auto',origin='lower',animated=True)
                im2.set_clim(np.min(np.angle(m)),np.max(np.angle(m)))
                plt.colorbar(im2,ax=ax2)
               
        else:
            ax1 = a[0]
            extent = [np.min(c), np.max(c), np.min(r), np.max(r)] 
            im1 = ax1.imshow(np.abs(m),extent=extent,aspect='auto',origin='lower',animated=True)
            im1.set_clim(np.min(np.abs(m)),np.max(np.abs(m)))
            plt.colorbar(im1,ax=ax1)

            if np.angle(m).any():
                ax2 = a[1]
                extent = [np.min(c), np.max(c), np.min(r), np.max(r)] 
                im2 = ax2.imshow(np.angle(m),cmap='hsv',vmin=-np.pi,vmax=np.pi,extent=extent,aspect='auto',origin='lower',animated=True)
                im2.set_clim(np.min(np.angle(m)),np.max(np.angle(m)))
                plt.colorbar(im2,ax=ax2)

        plt.show()

    def CurvePicture(self,kind,tag=None):
        
        data = np.load(''.join((self.pathaddress, '.npz')))
        r = np.abs(data['row'])
        c = np.asarray(data['col'])

        fig = plt.gcf()
        a = plt.getp(fig,'axes')

        ax1 = a[0]
        if tag:
            popt, pcov = tf.analyze(r,np.abs(c))
            A, B, T1 = popt 
            z = A * np.exp(-r / T1)+ B
            ax1.set_title(r'$T1 = %g \mathrm{ns}$' %T1)
            ax1.plot(r,z,'-')
        ax1.plot(r,np.abs(c),kind)
        ax1.grid()
        if np.angle(c).any():
            ax2 = a[1]
            ax2.grid()
            ax2.plot(r,np.angle(c))
        plt.show()

    def DyColor(self,tag=None):

        def UpdateValue():
            global path_str 
            while True:
                if self.q.poll():
                    path_str = self.q.recv()
                #if os.path.exists(''.join((pathaddress, '.npz'))):
                
                if path_str:
                    #if not os.path.exists(path_str[0]):
                    try:
                        
                            #with open(path_str[1],'rb') as pa:
                        #data = np.load(path_str[1],'rb')
                        data = path_str
                        #r = np.asarray(data['row'])
                        #c = np.asarray(data['col'])
                        #m = np.asarray(data['s'])
                        r = np.asarray(data[0])
                        c = np.asarray(data[1])
                        m = np.asarray(data[2])
                    except EOFError:
                        pass
                    yield r,c,m
        
        fig = plt.gcf()
        a = plt.getp(fig,'axes')
       
        xdata, ydata, zdata = next(UpdateValue())
        if np.min(xdata)==np.max(xdata):
            x = xdata
            y = xdata+0.001
            xdata = np.hstack((x,y))
        if np.min(ydata)==np.max(ydata):
            x = ydata
            y = ydata+0.001
            ydata = np.hstack((x,y))

        if tag =='db':
            zdata1 = 20*np.log10(np.abs(zdata))
            ax1 = a[0]
            extent = [np.min(ydata), np.max(ydata), np.min(xdata), np.max(xdata)] 
            im1 = ax1.imshow(zdata1,interpolation='nearest',aspect='auto',origin='lower',animated=True)
            im1.set_clim(np.min(zdata1),np.max(zdata1))
            im1.set_extent(extent)
       
            if np.angle(zdata).any():
                ax2 = a[1]
                im2 = ax2.imshow(np.angle(zdata),vmin=-np.pi,vmax=np.pi,interpolation='nearest',aspect='auto',origin='lower',animated=True)
                im2.set_clim(np.min(np.angle(zdata)),np.max(np.angle(zdata)))
                im2.set_extent(extent)
        else:
            ax1 = a[0]
            extent = [np.min(ydata), np.max(ydata), np.min(xdata), np.max(xdata)] 
            im1 = ax1.imshow(np.abs(zdata),interpolation='nearest',aspect='auto',origin='lower',animated=True)
            im1.set_clim(np.min(np.abs(zdata)),np.max(np.abs(zdata)))
            im1.set_extent(extent)
       
            if np.angle(zdata).any():
                ax2 = a[1]
                im2 = ax2.imshow(np.angle(zdata),vmin=-np.pi,vmax=np.pi,interpolation='nearest',aspect='auto',origin='lower',animated=True)
                im2.set_clim(np.min(np.angle(zdata)),np.max(np.angle(zdata)))
                im2.set_extent(extent)
 
        def CPicture(data):
            
            xdata, ydata, zdata = data
            if np.min(xdata)==np.max(xdata):
                x = xdata
                y = xdata+0.001
                xdata = np.hstack((x,y))
            if np.min(ydata)==np.max(ydata):
                x = ydata
                y = ydata+0.001
                ydata = np.hstack((x,y))
            if tag =='db':
                zdata1 = 20*np.log10(np.abs(zdata))
                extent = [np.min(ydata), np.max(ydata), np.min(xdata), np.max(xdata)] 
                im1.set_array(zdata1)
                im1.set_clim(np.min(zdata1),np.max(zdata1))
                im1.set_extent(extent)
                if np.angle(zdata).any():
                    im2.set_array(np.angle(zdata))
                    im2.set_clim(np.min(np.angle(zdata)),np.max(np.angle(zdata)))
                    im2.set_extent(extent)
            else:
                extent = [np.min(ydata), np.max(ydata), np.min(xdata), np.max(xdata)] 
                im1.set_array(np.abs(zdata))
                im1.set_clim(np.min(np.abs(zdata)),np.max(np.abs(zdata)))
                im1.set_extent(extent)
                if np.angle(zdata).any():
                    im2.set_array(np.angle(zdata))
                    im2.set_clim(np.min(np.angle(zdata)),np.max(np.angle(zdata)))
                    im2.set_extent(extent)
      
            return im1,im2

        ani = FuncAnimation(fig,CPicture,frames=UpdateValue,interval=50,blit=False)
        plt.show()
    
    def DyCurve(self,kind,tag=None):

        def UpdateValue():
            global path_str
            while True: 
                if self.q.poll():
                    path_str = self.q.recv()
                #if os.path.exists(''.join((pathaddress, '.npz'))):
                if path_str:
                    #if not os.path.exists(path_str[0]):
                    try:
                            #with open(path_str[1],'rb') as pa:
                        #data = np.load(path_str[1],'rb')
                        #r = np.abs(data['row'][()])
                        #c = np.asarray(data['col'][()])
                        data = path_str  
                        r = np.array(data[0])
                        c = np.asarray(data[1])
                    except EOFError:
                        pass
       
                    yield r,c

        fig = plt.gcf() 
        a = plt.getp(fig,'axes') 

        r, c = next(UpdateValue())
        ax1 = a[0]
        ax1.grid()
        line = ax1.plot([],[],kind,[],[],'b-')
        if np.angle(c).any():
            ax2 = a[1]
            ax2.grid()
            line2, = ax2.plot([],[])
        xdata1, ydata1 = [], []
        
        def init():
            ax1.set_ylim(-0.01, 0.01)
            ax1.set_xlim(0, 10)
            del xdata1[:]
            del ydata1[:]
            line[0].set_data(xdata1,ydata1)
            line[1].set_data(xdata1,ydata1)
            l = []
            if np.angle(c).any():
                ax1.set_ylim(-0.01, 0.01)
                ax1.set_xlim(0, 10)
                del xdata1[:]
                del ydata1[:]
                line2.set_data(xdata1,ydata1)
                l = []
            return line,l
        
        def CPicture(data):
            t, s = data
            xdata = np.array(t)
            ydata = np.abs(s)
            if np.min(xdata)==np.max(xdata):
                x = xdata
                y = xdata+0.001
                xdata = np.hstack((x,y))
            if np.min(ydata)==np.max(ydata):
                x = ydata
                y = ydata+0.001
                ydata = np.hstack((x,y))
            #if max(xdata) >= xmax:
            ax1.set_xlim(np.min(xdata), np.max(xdata))
            ax1.figure.canvas.draw()
            #if max(ydata) >= ymax:
            ax1.set_ylim(np.min(ydata),np.max(ydata))
            ax1.figure.canvas.draw()
            if tag:
                try:
                    popt, pcov = tf.analyze(xdata,ydata)
                    A, B, T1 = popt
                    z =  A * np.exp(-xdata/T1) + B
                    line[0].set_data(xdata,ydata)
                    line[1].set_data(xdata,z)
                    #ax.add_line(line[1])
                    ax1.set_title(r'$T1 = %g \mathrm{ns}$' %T1)
                except RuntimeError or TypeError:
                    pass
                
            else:
                line[0].set_data(xdata,ydata)
            l = []
            if np.angle(s).any():
                if np.min(np.angle(s))==np.max(np.angle(s)):
                    x = s
                    y = s+1
                    s = np.hstack((x,y))
                #if max(t) >= xmax:
                ax2.set_xlim(np.min(xdata),np.max(xdata))
                ax2.figure.canvas.draw()
                #if max(np.angle(y)) >= np.max(np.angle(y)):
                ax2.set_ylim(np.min(np.angle(s)),np.max(np.angle(s)))
                ax2.figure.canvas.draw()
                line2.set_data(t,np.angle(s))
                l = line2
            return line,l


        ani = FuncAnimation(fig,CPicture,frames=UpdateValue,init_func=init,interval=50,blit=False)
        plt.show()