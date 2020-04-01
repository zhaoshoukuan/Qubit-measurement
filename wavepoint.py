#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wavepoint.py
@Time    :   2019/08/27 23:22:23
@Author  :   sk zhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib
import numpy as np

class WAVE_FORM():

    def __init__(self,t_list):
        self.t = t_list

    def step(self,t_end,width):
        return (self.t>(t_end-width))*1-(self.t>t_end)*1
    
    def wave(self,t_end=0,width=0,height=0,tag=None):
        pulse_want = 0
        for i in range(len(t_end)):
            if tag=='+-':
                pulse_want += (self.step(t_end[i],width[i]))*height[i]
                pulse_want = pulse_want*2-1
            else:
                pulse_want += self.step(t_end[i],width[i])*height[i]
        return pulse_want

    #waveform
    def square_wave(self,t_end=0,width=0,height=0,tag=None):
        pulse_want = self.wave(t_end,width,height,tag)
        return pulse_want
        
    def sin_envelope(self,f=0,phi=0,t_end=0,width=0,height=0):
        pulse_want = 0
        for i in range(len(t_end)):
            pulse_single = 0
            for j in range(len(f[i])):
                pulse_single += np.sin(2*np.pi*f[i][j]*1e-9*self.t+phi[i][j])
                pulse_single = pulse_single / len(f[i])
            pulse_want += self.wave(t_end[i],width[i],height[i])*1/2*(np.cos(2*np.pi/width[i][0]*(self.t-t_end[i][0]+width[i][0]/2))+1)*pulse_single
        #pulse_want = self.wave(t_end,width,height,tag)*1/2*(np.cos(2*np.pi/width[0]*(t-t_end[0]+width[0]/2))+1)*np.sin(2*np.pi*f[0]*1e-9*t+phi[0])
        return pulse_want
        
    def square_envelope(self,f=0,phi=0,t_end=0,width=0,height=0):
        pulse_want = 0
        for i in range(len(t_end)):
            pulse_single = 0
            for j in range(len(f[i])):
                pulse_single += np.sin(2*np.pi*f[i][j]*1e-9*self.t+phi[i][j])
            pulse_single = pulse_single / len(f[i])
            pulse_want += self.wave(t_end[i],width[i],height[i])*pulse_single
        #pulse_want = self.wave(t_end,width,height,tag)*np.sin(2*np.pi*f[0]*1e-9*t+phi)
        return pulse_want

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t_list = np.linspace(0,100000,250000)
    wf = WAVE_FORM(t_list)
    f, phi, t_end, width, height = [50e6,60e6], [0]*2, [91000], [1000], [1]
    s = wf.square_envelope(f,phi,t_end,width,height)
    plt.plot(t_list,s)
    plt.show()
