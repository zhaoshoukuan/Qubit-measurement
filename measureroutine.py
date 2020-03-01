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
import numpy as np, serial, time
from qulab.job import Job
from qulab.wavepoint import WAVE_FORM as WF
from collections import Iterable
from tqdm import tqdm_notebook as tqdm
from qulab.optimize import Collect_Waveform 

t_list = np.linspace(0,100000,250000)
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

class common():
    def __init__(self,freqall,ats,dc,psg,awg,jpa):
        self.freqall = freqall
        self.ats = ats
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.jpa = jpa
        self.wave = {}

class qubitCollections():
    def __init__(self,qubits,q_target=None):
        self.qubits = {i.q_name:i for i in qubits}
        self.f_lo = np.array([i.f_lo[0] for i in qubits])
        self.f_ex = np.array([i.f_ex[0] for i in qubits])
        if q_target != None:
            self.inst = self.qubits[q_target]
            for i in qubits:
                if i.q_name == q_target:
                    self.q = i 
                    self.q_name = i.q_name
        else:
            self.inst = {i.q_name:i.inst for i in qubits}
    def qubitExecute(self,q_target=[]):
        q = []
        for i in q_target:
            q.append(self.qubits[i])
        return q

################################################################################
### 设置衰减器
################################################################################

class Att_Setup():
    
    def __init__(self,com):
        self.com = com
        ser = serial.Serial(self.com,baudrate=115200, parity='N',bytesize=8, stopbits=1, timeout=1)
        self.ser = ser
        if ser.isOpen():    # make sure port is open     
            print(ser.name + ' open...')
            ser.write(b'*IDN?\n')
            x = ser.readline().decode().split('\r''\n')
            print(x[0])
            ser.write(b'ATT?\n')
            y = ser.readline().decode().split('\r''\n')
            print('ATT',y[0])

    def Att(self,att):

        self.ser.write(b'ATT %f\n'%att)
        time.sleep(1)
        self.ser.write(b'ATT?\n')

    def close(self):

        self.ser.close()
        
################################################################################
### 设置采集卡
################################################################################

async def ats_setup(ats,delta,l=1000,repeats=500,awg=0):
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

async def modulation_read(ats,awg,delta,tdelay=1000):
    global t_list
    twidth, n = tdelay, len(delta)
    wavelen = (twidth // 64) * 64
    if wavelen < twidth:
        wavelen += 64
    f, phi, t_end, width, height = [delta], [[0]*n], [[90000+wavelen]], [[wavelen]], [[1]]
    wf = WF(t_list)
    await awg.create_waveform(name='Readout_Q', length=len(t_list), format=None)
    await awg.create_waveform(name='Readout_I', length=len(t_list), format=None)
    sample = wf.square_envelope(f,phi,t_end,width,height)
    await awg.update_waveform(sample/np.abs(sample).max(),name='Readout_Q')
    
    f, phi, t_end, width, height = [delta], [[np.pi/2]*n], [[90000+wavelen]], [[wavelen]], [[1]]
    sample = wf.square_envelope(f,phi,t_end,width,height)
    await awg.update_waveform(sample/np.abs(sample).max(),name='Readout_I')
    
    sample = wf.square_wave(t_end=[90145+128/2+wavelen],width=[wavelen],height=[1])
    await awg.update_marker(name='Readout_Q',mk1=sample)
    await awg.update_marker(name='Readout_I',mk1=sample)
    await ats_setup(ats,delta,l=tdelay)
    time.sleep(5)
    
################################################################################
### 激励混频
################################################################################

async def modulation_ex(qubit,measure,w=25000,delta_ex=[0]):
    global t_list
    wf, Iname, Qname, n = WF(t_list), qubit.q_name+'_I', qubit.q_name+'_Q', len(delta_ex)
    await measure.awg[qubit.inst['ex_awg']].create_waveform(name=Qname, length=len(t_list), format=None)
    await measure.awg[qubit.inst['ex_awg']].create_waveform(name=Iname, length=len(t_list), format=None)

    f,phi,height,t_end,width = delta_ex,[0]*n,[1],[89700],[w]
    sample = wf.square_wave(t_end,width,height)
    await measure.awg[qubit.inst['ex_awg']].update_waveform(sample,name=Qname)
    await measure.awg[qubit.inst['ex_awg']].update_marker(name=Qname,mk1=sample)
    f,phi,height,t_end,width = delta_ex,[np.pi/2]*n,[1],[89700],[w]
    sample = wf.square_wave(t_end,width,height)
    await measure.awg[qubit.inst['ex_awg']].update_waveform(sample,name=Iname)

    await measure.awg[qubit.inst['ex_awg']].use_waveform(name=Iname,ch=qubit.inst['ex_ch'][0])
    await measure.awg[qubit.inst['ex_awg']].use_waveform(name=Qname,ch=qubit.inst['ex_ch'][1])

    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][0])
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][1])
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
### 开关JPA
################################################################################

async def jpa_switch(measure,state='OFF'):
    if state == 'ON':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','ON')
        await measure.psg[measure.jpa.inst['pump']].setValue('Frequency',(measure.jpa.f_ex))
        await measure.psg[measure.jpa.inst['pump']].setValue('Power',measure.jpa.power_ex)
        await measure.dc[measure.jpa.inst['dc']].DC(measure.jpa.bias)
    if state == 'OFF':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','OFF')
        await  measure.dc[measure.jpa.inst['dc']].DC(0)

################################################################################
### 关闭仪器
################################################################################

async def close(measure):
    inst = {**measure.dc , **measure.psg}
    for i in inst:
        await inst[i].setValue('Output','OFF')

################################################################################
### 查询仪器状态
################################################################################

async def QueryInst(measure):
    inst = {**measure.dc , **measure.psg}
    state = {}
    for i in inst:
        try:
            current = await inst[i].getValue('Offset')
            if current != None:
                load = await inst[i].getValue('Load')
                load = eval((load).strip('\n'))  
                load = 'high Z' if load != 50 else 50
                sm = {'offset':current,'load':load}
                state[i] = sm
            else:
                freq = await inst[i].getValue('Frequency')
                power = await inst[i].getValue('Power')
                Output = await inst[i].getValue('Output')
                Moutput = await inst[i].getValue('Moutput')
                sm = {'freq':'%fGHz'%(freq/1e9),'power':'%fdBm'%power,'output':Output,'moutput':Moutput}
                state[i] = sm
        finally:
            pass
    return state

################################################################################
### S21
################################################################################

async def S21(qubit,measure,modulation=False,freq=None):
    #await jpa_switch(measure,state='OFF')
    if freq == None:
        f_lo, delta, n = await resn(np.array(qubit.f_lo))
        freq = np.linspace(-2.5,2.5,126)*1e6 + f_lo
        if modulation:
            await modulation_read(measure.ats,measure.awg['awg133'],delta)
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

async def again(qubit,measure,modulation=False):
    #f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    #freq = np.linspace(-2.5,2.5,126)*1e6+f_lo
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','OFF')
    job = Job(S21, (qubit,measure,modulation),auto_save=False,max=126)
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
   
    return f_lo, delta, n, f_res, base

################################################################################
### S21vsFlux
################################################################################

async def S21vsFlux(qubit,measure,current,modulation=False):
    for i in current:
        await measure.dc[qubit.inst['dc']].DC(i)
        job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        yield [i]*1, f_s21, s_s21

################################################################################
### SingleSpec
################################################################################

async def singlespec(qubit,measure,freq,modulation=False):
    await jpa_switch(measure,'ON')
    await modulation_ex(qubit,measure)
    f_lo, delta, n, f_res,base = await again(qubit,measure,modulation)
    measure.base, measure.n = base, n
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    for i in freq:
        await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',i)
        ch_A, ch_B = await measure.ats.getIQ(hilbert=True,offset=True)
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base

################################################################################
### Spec2d
################################################################################

async def spec2d(qubit,measure,freq):
    f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    current = np.linspace(-qubit.T_bias[0]*0.3,qubit.T_bias[0]*0.3,31) + qubit.T_bias[1] 
    await jpa_switch(measure,'ON')
    
    for i in current:
        await measure.dc[qubit.inst['dc']].DC(i)
        f_lo, delta, n, f_res, base = await again(qubit)
        await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
        job = Job(singlespec, (qubit,measure,freq),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        yield [i]*n, f_ss, s_ss

################################################################################
### 创建波形及sequence
################################################################################

async def create_wavelist(measure,kind,para):
    awg,name,n_wave,length = para
    @Collect_Waveform(measure.wave,kind)
    async def create_waveformlist(awg,name,n_wave,length):
        name_list = []
        for i in tqdm(name,desc='create_waveformlist'):
            name_collect = []
            for j in range(1,n_wave+1):
                name_w = ''.join((i,'_%d'%j))
                name_collect.append(name_w)
                await measure.awg[awg].create_waveform(name=name_w, length=length, format=None) 
            name_list.append(name_collect)
        return name_list
    return create_waveformlist(*para)

################################################################################
### Rabi
################################################################################

async def rabi(qubit,measure,t,len_data):
    t = t[1:len_data+1]
    t = np.array([t]*measure.n).T
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ(hilbert=True)
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        theta0 = np.angle(Am) - np.angle(Bm)
        Bm *= np.exp(1j*theta0)
        s = Am + Bm
        yield t, s - measure.base