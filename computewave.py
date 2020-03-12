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
import numpy as np, time, pickle
from qulab.wavepoint import WAVE_FORM as WF
from qulab.waveform import CosPulseDrag, Expi, DC, Step, Gaussian, CosPulse
from tqdm import tqdm_notebook as tqdm
from qulab.optimize import Collect_Waveform 
from qulab import imatrix as mx


t_list = np.linspace(0,100000,250000)
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

################################################################################
### 收集变量
################################################################################

def saveStatus(fname='D:/status.obj'):
    import sys
    
    status = {}
    
    for k in filter(
        lambda s: s[0] != '_' and not callable(globals()[s]) and
        not isinstance(globals()[s], type(sys)) and not s in ['In', 'Out'],
        globals().keys()):
        try:
            status[k] = pickle.dumps(globals()[k])
        except:
            print(k, type(globals()[k]))
            
    with open(fname, 'wb') as f:
         pickle.dump(status, f)
    
def loadStatus(fname='D:/status.obj'):
    with open(fname, 'rb') as f:
        status = pickle.load(f)
    
    for k, v in status.items():
        globals()[k] = pickle.loads(v)

################################################################################
### 创建波形及sequence
################################################################################

async def create_wavelist(measure,kind,para):
    awg,name,n_wave,length = para
    if kind in measure.wave:
        print('kind has already existed')
        return
    @Collect_Waveform(measure.wave,kind)
    async def create_waveformlist(awg,name,n_wave,length):
        name_list = []
        for i in tqdm(name,desc='create_waveformlist'):
            name_collect = []
            for j in range(1,n_wave+1):
                name_w = ''.join((kind,'_',i,'%d'%j))
                name_collect.append(name_w)
                await measure.awg[awg].create_waveform(name=name_w, length=length, format=None) 
            name_list.append(name_collect)
        return name_list
    return create_waveformlist(*para)

################################################################################
### 生成Sequence
################################################################################

async def genSeq(measure,awg,kind):
    await awg.create_sequence(kind,len(measure.wave[kind][0])+1,8)
    for j,i in enumerate(measure.wave[kind],start=1):
         await awg.set_seq(i,1,j,seq_name=kind)

################################################################################
### 生成Sequence
################################################################################

async def readSeq(measure,awg,kind):
    await awg.create_sequence(kind,len(measure.wave[kind][0])+1,8)
    for j,i in enumerate(measure.wave[kind],start=1):
         await awg.set_seq(i,1,j,seq_name=kind)
    await measure.awg['awgread'].use_sequence(kind,channels=[7,8])
    await measure.awg['awgread'].query('*OPC?')
    await measure.awg['awgread'].output_on(ch=7)
    await measure.awg['awgread'].output_on(ch=8)
    await measure.awg['awgread'].run()
    await measure.awg['awgread'].query('*OPC?')
################################################################################
### Rabi波形
################################################################################

async def rabiWave(awg,during=75e-9, shift=200e-9, Delta_lo=80e6,name=['Ex_I','Ex_Q']):
    
    lo = Expi(2*np.pi*Delta_lo)
    m1 = CosPulse(during) << during/2
    m2 = CosPulse(during) << during*3/2
    m = m1 + m2
    wav = (m << shift) * lo

    # lo = Expi(2*np.pi*Delta_lo)
    # m1 = (Step(2e-9)<<during) - Step(2e-9)
    # m2 = ((Step(2e-9)<<during) - Step(2e-9)) << during
    # wav = ((m1 + m2) << shift) * lo

    # lo = Expi(2*np.pi*Delta_lo)
    # init = (Step(2e-9)<<during) - Step(2e-9)
    # wav = (init << shift) * lo

    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    I, Q = np.real(points), np.imag(points)
    await awg.update_waveform(I, name = name[0])
    await awg.update_waveform(Q, name = name[1])

    init1 = (Step(0e-9)<<during) - Step(0e-9)
    wav1 = (init1 << shift) 
    wav1.set_range(*t_range)
    points1 = wav1.generateData(sample_rate)
    await awg.update_marker(name=name[1],mk1=points1)
    #await awg.update_marker(name[1],mk1=points1)

async def Rabi_sequence(measure,kind,awg,t_rabi):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in enumerate(tqdm(t_rabi,desc='Rabi_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rabiWave(measure.awg[awg],during=i/1e9, name = name_ch)
    await measure.awg[awg].query('*OPC?')

################################################################################
### T1波形
################################################################################

async def T1_sequence(measure,kind,awg,t_rabi,pi_len):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in enumerate(tqdm(t_rabi,desc='T1_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rabiWave(measure.awg[awg],during=pi_len/1e9,shift=(i+200)*1e-9, name = name_ch)
    await measure.awg[awg].query('*OPC?')

################################################################################
### Ramsey波形
################################################################################

async def ramseyWave(awg,delay,halfpi=75e-9,fdetune=3e6, shift=200e-9, Delta_lo=80e6,name=['Ex_I','Ex_Q']):
    
    lo2, lo1 = Expi(2*np.pi*Delta_lo), Expi(2*np.pi*Delta_lo,2*np.pi*fdetune*delay)
    m1 = (CosPulse(halfpi) << halfpi/2) * lo1 
    m2 = (CosPulse(halfpi) << (halfpi*3/2 + delay)) * lo2  #m1,m2,lo1,lo2所乘位置要考察
    wav = (m1 + m2) << shift
    # lo2, lo1 = Expi(2*np.pi*Delta_lo), Expi(2*np.pi*Delta_lo,2*np.pi*fdetune*delay)
    # m1 = (Step(2e-9)<<halfpi) - Step(2e-9) *lo1
    # m2 = (((Step(2e-9)<<halfpi) - Step(2e-9)) << (halfpi + delay)) * lo2
    # wav = (m1 + m2) << shift
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    I, Q = np.real(points), np.imag(points)
    await awg.update_waveform(I, name = name[0])
    await awg.update_waveform(Q, name = name[1])
    #await awg.update_marker(name[1],mk1=points1)

async def Ramsey_sequence(measure,kind,awg,t_rabi,halfpi):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg].stop()
    time.sleep(5)
    for j,i in enumerate(tqdm(t_rabi,desc='Ramsey_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await ramseyWave(measure.awg[awg], delay=i/1e9, halfpi=halfpi/1e9, name = name_ch)
    await measure.awg[awg].query('*OPC?')

################################################################################
### Z_cross波形
################################################################################

async def z_crossWave(awg,volt,halfpi,during=300e-9, shift=200e-9,name=['q1_z']):

    init = ((Step(2e-9)<<during) - Step(2e-9)) * volt
    wav = (init << (shift+halfpi+75e-9)) 
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    
    I, Q = np.real(points), np.imag(points)

    await awg.update_waveform(I, name[0])
    #await awg.update_waveform(Q, name[1])
    
async def Z_cross_sequence(measure,kind_z,kind_ex,awg_z,awg_ex,v_rabi,halfpi):
    #t_range, sample_rate = measure.t_range, measure.sample_rate
    await measure.awg[awg_ex].stop()
    await measure.awg[awg_z].stop()
    time.sleep(5)
    for j,i in enumerate(tqdm(v_rabi,desc='Z_cross_sequence')):
        name_ch = [measure.wave[kind_ex][0][j],measure.wave[kind_ex][1][j]]
        await ramseyWave(measure.awg[awg_ex], delay=450/1e9, halfpi=halfpi/1e9, fdetune=0, name = name_ch)
        await z_crossWave(measure.awg[awg_z],volt=i,halfpi=halfpi/1e9,name=[measure.wave[kind_z][0][j]])
    await measure.awg[awg_ex].query('*OPC?')
    await measure.awg[awg_z].query('*OPC?')

################################################################################
### AC-Stark波形
################################################################################

async def ac_stark_sequence(measure,awg,kind,pilen,t_rabi):
    awg_read, awg_ex = measure.awg['awgread'], measure.awg[awg]
    await awg_ex.stop()
    await awg_read.stop() 
    time.sleep(5)
    lo = Expi(2*np.pi*50e6)
    m1 = (Step(0e-9)<<measure.wavelen*1e-9) - Step(0e-9) >> measure.wavelen*1e-9
    m2 = ((Step(0e-9)<<measure.wavelen*1e-9) - Step(0e-9)) << 500e-9
    wav = (m1 + m2) * lo
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    I, Q = np.real(points), np.imag(points)
    await awg_read.update_waveform(I, name = 'Readout_I')
    await awg_read.update_waveform(Q, name = 'Readout_Q')

    for j,i in enumerate(tqdm(t_rabi,desc='acStark_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rabiWave(awg_ex,during=pilen/1e9,shift=i*1e-9, name = name_ch)
    await awg_ex.query('*OPC?')

################################################################################
### RB波形
################################################################################

def genXY(phi,during,height=1,shift=0e-9,tdelay=0e-9,Delta_lo=80e6,pulse='pi'):
    shift += 200e-9
    lo = Expi(2*np.pi*Delta_lo,phi)
    if pulse == 'pi':
        m1 = CosPulse(during) << during/2
        m2 = CosPulse(during) << (during*3/2+tdelay)
        m = m1 + m2
    if pulse == 'halfpi':
        m = CosPulse(during) << during/2
    wav = (m*height << shift) * lo
        
    return wav

def genParas(x,pilen):
    if x == 'I':
        paras = (0,0,'pi')
    elif x == 'X':
        paras = (0,1,'pi')
    elif x == 'Xhalf':
        paras = (0,1,'halfpi')
    elif x == 'Xnhalf':
        paras = (np.pi,1,'halfpi')
    elif x == 'Y':
        paras = (np.pi/2,1,'pi')
    elif x == 'Yhalf':
        paras = (np.pi/2,1,'halfpi')
    elif x == 'Ynhalf':
        paras = (3*np.pi/2,1,'halfpi')
    return paras

async def rbWave(awg,m,pilen,name=['Ex_I','Ex_Q']):
    op = {'1':['I'],'2':['X'],'3':['Xhalf'],'4':['Xnhalf'],'5':['Y'],'6':['Yhalf'],'7':['Ynhalf'],
        '8':['X','Y'],'9':['Xhalf','Yhalf','Xnhalf'],'10':['Xhalf','Ynhalf','Xnhalf'],'11':['Ynhalf','X'],
        '12':['Yhalf','X'],'13':['Xhalf','Y'],'14':['Xnhalf','Y'],'15':['Xhalf','Yhalf','Xhalf'],'16':['Xnhalf','Yhalf','Xnhalf'],
        '17':['Xhalf','Yhalf'],'18':['Xnhalf','Yhalf'],'19':['Xhalf','Ynhalf'],'20':['Xnhalf','Ynhalf'],
        '21':['Ynhalf','Xnhalf'],'22':['Ynhalf','Xhalf'],'23':['Yhalf','Xnhalf'],'24':['Yhalf','Xhalf']}

    mseq, invertseq = mx.cliffordGroup_single(m)
    rotseq = []
    for i in mseq:
        rotseq += op[i]
    rotseq.append(invertseq)
    shift, waveseq = 0, 0
    for i in rotseq[::-1]:
        paras = genParas(i,pilen)
        wav = genXY(phi=paras[0],during=pilen,height=paras[1],pulse=paras[2])
        waveseq += wav << shift
        if paras[2] == 'pi':
            shift += 2*pilen
        if paras[2] == 'halfpi':
            shift += pilen

    waveseq.set_range(*t_range)
    points = waveseq.generateData(sample_rate)
    I, Q = np.real(points), np.imag(points)
    await awg.update_waveform(I, name[0])
    await awg.update_waveform(Q, name[1])
    
async def rb_sequence(awg,kind,mlist,pilen):
    await awg.stop()
    await awg.query('*OPC?')
    for j,i in enumerate(tqdm(mlist,desc='RB_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await rbWave(awg,i,pilen*1e-9,name=name_ch)
    await awg.query('*OPC?')



