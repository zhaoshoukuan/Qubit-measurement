#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   measure_routine.py
@Time    :   2020/02/20 10:24:45
@Author  :   sk zhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib
import numpy as np, time
from qulab.job import Job
from qulab.wavepoint import WAVE_FORM as WF


class common():
    def __init__(self,freqall,ats,dc,psg,awg):
        self.freqall = freqall
        self.ats = ats
        self.dc = dc
        self.psg = psg
        self.awg = awg

################################################################################
### 设置采集卡
################################################################################

async def ats_setup(ats,delta,repeats=500,awg=0):
    l = 1024
    await ats.set(n=l,repeats=repeats,awg=awg,
                           f_list=delta,
                           maxlen=l,
                           ARange=1.0,
                           BRange=1.0,
                           trigLevel=0.5,
                           triggerDelay=0,
                           triggerTimeout=0,
                           bufferCount=l)
    
################################################################################
### 读出混频
################################################################################

async def modulation_read(awg,delta,n):
    t_list = np.linspace(0,100000,250000)

    f, phi, t_end, width, height = [delta], [[0]*n], [[91000]], [[1000]], [[1]]
    wf = WF(t_list)
    await awg.create_waveform(name='Readout_Q', length=len(t_list), format=None)
    await awg.create_waveform(name='Readout_I', length=len(t_list), format=None)
    sample = wf.square_envelope(f,phi,t_end,width,height)
    await awg.update_waveform(sample/np.abs(sample).max(),name='Readout_Q')
    
    f, phi, t_end, width, height = [delta], [[np.pi/2]*n], [[91000]], [[1000]], [[1]]
    sample = wf.square_envelope(f,phi,t_end,width,height)
    await awg.update_waveform(sample/np.abs(sample).max(),name='Readout_I')
    
    sample = wf.square_wave(t_end=[91115],width=[1000],height=[1])
    await awg.update_marker(name='Readout_Q',mk1=sample)
    await awg.update_marker(name='Readout_I',mk1=sample)
    time.sleep(5)
    
################################################################################
### 腔频率设置
################################################################################

async def resn(f_cavity):
    f_lo = f_cavity.max() + 50e6
    delta =  f_lo - f_cavity 
    n = len(f_cavity)
    return f_lo, delta, n

################################################################################
### S21
################################################################################

async def S21(qubit,measure,freq=None):
    if freq == None:
        freq, delta = np.linspace(-4,4,81)*1e6 + qubit.f_lo, qubit.delta
    await measure.psg['psg_lo'].setValue('Output','ON')
    for i in freq:
        await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B = await measure.ats.getIQ(hilbert=True)
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield i-delta, s

################################################################################
### 重新混频
################################################################################

async def again(qubit,measure):
    f_lo, delta, n = qubit.f_lo, qubit.delta, 1
    freq = np.linspace(-4,4,161)*1e6+f_lo
    await measure.psg['psg_ex'].setValue('Output','OFF')
    job = Job(S21, (qubit,measure),auto_save=False,max=len(freq))
    f_s21, s_s21 = await job.done()
    index = np.abs(s_s21).argmin(axis=0)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n = await resn(np.array(f_res))
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        await ats_setup(measure.ats,delta)
        await modulation_read(measure.awg['wg133'],delta,n)
        base = 0
        for i in range(50):
            ch_A, ch_B = await measure.ats.getIQ(hilbert=True)
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            theta0 = np.angle(Am) - np.angle(Bm)
            Bm *= np.exp(1j*theta0)
            base += Am + Bm
        base /= 50
   
    return f_lo, delta, n, f_res,base

################################################################################
### S21vsFlux
################################################################################

async def S21vsFlux(qubit,measure,current):
    for i in current:
        await measure.dc[qubit.inst['dc']].DC(i)
        job = Job(S21, (qubit,measure),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        yield [i]*1, f_s21, s_s21