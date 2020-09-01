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
from easydl import clear_output
from qulab.wavepoint import WAVE_FORM as WF
from qulab.waveform import CosPulseDrag, Expi, DC, Step, Gaussian, CosPulse
from qulab import waveform_new as wn
from tqdm import tqdm_notebook as tqdm
import asyncio, inspect
from qulab import imatrix as mx


t_list = np.linspace(0,100000,250000)
t_new = np.linspace(-90000,10000,250000)*1e-9
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

################################################################################
# 收集变量
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
# 函数参数解析及赋值
################################################################################

async def funcarg(f,config,**kws):
    # cp = ConfigParser()
    # config.read(r'D:\QuLabData\config\status.ini')
    section = f.__name__
    insp = inspect.getfullargspec(f)
    paras, defaults = insp[0], insp[3]
    status = {}
    for i in paras:
        if i in config:
            status[i] = eval(config.get(section,i))
        else:
            status[i] = kws[i]
    await f(**status)
    
################################################################################
# 收集awg波形名字
################################################################################

def Collect_Waveform(dictname,kind):
    
    def decorator(func):
        def wrapper(*args, **kw):
            if asyncio.iscoroutinefunction(func):
                loop = asyncio.get_event_loop()
                name_list = loop.run_until_complete(func(*args, **kw))
                dictname[kind] = name_list
            else:
                return func(*args, **kw)
        return wrapper
    return decorator

################################################################################
# 创建波形及sequence
################################################################################

async def create_wavelist(measure,kind,para):
    awg,name,n_wave,length = para
    if kind in measure.wave:
        print('kind has already existed')
        return
    @Collect_Waveform(measure.wave,kind)
    async def create_waveformlist(awg,name,n_wave,length):
        name_list = []
        for i in name:
            name_collect = []
            for j in range(1,n_wave+1):
                name_w = ''.join((kind,'_',i,'%d'%j))
                name_collect.append(name_w)
                await awg.create_waveform(name=name_w, length=length, format=None) 
            name_list.append(name_collect)
        return name_list
    return create_waveformlist(*para)

################################################################################
# 询问AWG状态
################################################################################

async def couldRun(awg):
    await awg.run()
    # await awg.query('*OPC?')
    while True:
        time.sleep(0.5)
        x = await awg.query('AWGCONTROL:RSTATE?')
        if x == 1 or 2:
            break

async def clearSeq(measure,awg):
    for i in awg:
        await measure.awg[i].stop()
        await measure.awg[i].write('*WAI')
        for j in range(8):
            await measure.awg[i].output_off(ch=j+1)
        await measure.awg[i].clear_sequence_list()
    measure.wave = {}

################################################################################
# awg载入sequence
################################################################################

async def awgchmanage(awg,seqname,ch):
    await awg.stop()
    await awg.use_sequence(seqname,channels=ch)
    await awg.query('*OPC?')
    for i in ch:
        await awg.output_on(ch=i)
    await couldRun(awg)
    
################################################################################
# awg生成并载入waveform
################################################################################

async def genwaveform(awg,wavname,ch,t_list=t_list):
    # t_list = measure.t_list
    await awg.stop()
    for j, i in enumerate(wavname):
        await awg.create_waveform(name=i, length=len(t_list), format=None)
        await awg.use_waveform(name=i,ch=ch[j])
        await awg.write('*WAI')
        await awg.output_on(ch=ch[j])

################################################################################
# 生成Sequence
################################################################################

async def genSeq(measure,awg,kind):
    # length = await awg.query('SLISt:SEQuence:LENGth? "%s"'%kind)
    # if length > len(measure.wave[kind][0])+1:
    #     await awg.remove_sequence(kind)
    wait = 'ITR' if awg == measure.awg['awg_trig'] else 'ATR'
    await awg.create_sequence(kind,len(measure.wave[kind][0])+1,8)
    for j,i in enumerate(measure.wave[kind],start=1):
         await awg.set_seq(i,1,j,seq_name=kind,wait=wait)

################################################################################
# 生成读出Sequence
################################################################################

async def readSeq(measure,awg,kind,ch):
    # length = await awg.query('SLISt:SEQuence:LENGth? "%s"'%kind)
    # if length > len(measure.wave[kind][0])+1:
    #     await awg.remove_sequence(kind)
    wait = 'ITR' if awg == measure.awg['awg_trig'] else 'ATR'
    await awg.create_sequence(kind,len(measure.wave[kind][0])+1,8)
    for j,i in enumerate(measure.wave[kind],start=1):
         await awg.set_seq(i,1,j,seq_name=kind,wait=wait)
    await awgchmanage(awg,kind,ch)

################################################################################
# awg生成子sequence
################################################################################

async def subSeq(measure,awg,subkind,lensubseq):
    subkind = ''.join(('sub_',subkind))
    await create_wavelist(measure,subkind,(awg,['I','Q'],lensubseq,len(measure.t_list)))
    name = np.array(measure.wave[subkind])
    for i, j in enumerate(name,start=1):
        await awg.set_sequence_step(subkind,j,i,wait='ATR',goto='NEXT',repeat=1,jump=('OFF','NEXT'))
        
################################################################################
# awgDC波形
################################################################################

async def awgDC(awg,wavname,volt):
    await awg.stop()
    # init = DC(offset=-0.7, length=100e-6, range=(0,1))*0 + volt
    # wav = init
    init  = ((Step(2e-9)<<3000e-9) - Step(2e-9))*volt
    wav = (init >> 2250e-9)
    wav.set_range(*t_range)
    points = wav.generateData(sample_rate)
    await awg.update_waveform(points,wavname)

################################################################################
# Z波形矫正
################################################################################

def predistort(waveform, sRate, zCali):
    """Predistort input waveform.
    Parameters
    ----------
        waveform : complex numpy array
            Waveform data to be pre-distorted
        zCali: [2.55e-9, -28e-3, 8.02e-9, -28e-3, 101.5e-9, -14.5e-3,369.7e-9,-6.2e-3]
    Returns
    -------
    waveform : complex numpy array
        Pre-distorted waveform
    """
    dt = 1 / sRate
    wf_size = len(waveform)
    tau1, A1, tau2, A2, tau3, A3, tau4, A4, tau5, A5 = zCali
    # print('tau:', tau1, A1, tau2, A2, tau3, A3, tau4, A4, tau5, A5)
    # pad with zeros at end to make sure response has time to go to zero
    # pad_time = 6 * max([tau1, tau2, tau3])
    pad_time = 4e-6  # 这里默认改为增加4us的pad，返回时舍去1us，保留3us拖尾
    pad_size = round(pad_time / dt)
    pad_size_2 = round(3e-6 / dt)  # 保留的点数
    padded_zero = np.zeros(pad_size)
    padded = np.append(waveform,padded_zero)

    Y = np.fft.rfft(padded, norm='ortho')

    omega = 2 * np.pi * np.fft.rfftfreq(wf_size+pad_size, dt)
    H = (1 + (1j * A1 * omega * tau1) / (1j * omega * tau1 + 1) +
         (1j * A2 * omega * tau2) / (1j * omega * tau2 + 1) +
         (1j * A3 * omega * tau3) / (1j * omega * tau3 + 1) +
         (1j * A4 * omega * tau4) / (1j * omega * tau4 + 1)+
         (1j * A5 * omega * tau5) / (1j * omega * tau5 + 1))

    Yc = Y / H
    yc = np.fft.irfft(Yc, norm='ortho')
    # return yc[:wf_size]
    #这里返回增加了3us拖尾的序列
    return yc[:(wf_size+pad_size_2)]

################################################################################
# 更新波形
################################################################################

async def writeWave(awg,name,pulse,norm=False,t=t_new,mark=False):

    # t = np.linspace(-90000,10000,25000)*1e-9
    await awg.stop()
    if len(pulse) == 4:
        wav_I, wav_Q, mrk1, mrk2 = pulse
        I, Q = wav_I(t), wav_Q(t)
        # I, Q = pulse
        if norm:
            I, Q = I / np.max(np.abs(I)), Q / np.max(np.abs(Q))
        await awg.update_waveform(I, name = name[0])
        await awg.update_waveform(Q, name = name[1])
        if mark:
            # for i, j in enumerate(mark):
            mrk1 = [i(t) for i in mrk1]
            mrk2 = [i(t) for i in mrk2]
            await awg.update_marker(name[0],*mrk1)
            await awg.update_marker(name[1],*mrk2)
    if len(pulse) == 1:
        wave = pulse[0]
        # t = np.linspace(-90000,7000,242500)*1e-9
        # wave = pulse
        wave = wave(t)
        # wave = predistort(wave,2.5e9,[2.55e-9, -28e-3, 8.02e-9, -28e-3, 101.5e-9, -14.5e-3,369.7e-9,-6.2e-3,5e-9,-5e-3])
        await awg.update_waveform(wave, name = name[0])

################################################################################
# 波包选择
################################################################################

def whichEnvelope(envelop):
    x = {'square':wn.square,'hanning':wn.hanning,'hamming':wn.hamming,'gaussian':wn.gaussian}
    return x[envelop]

################################################################################
# 设置采集卡
################################################################################

async def ats_setup(ats,delta,l=1000,repeats=500,awg=0):
    await ats.set(n=l,repeats=repeats,awg=awg,
                           f_list=delta,
                           maxlen=512,
                           ARange=1.0,
                           BRange=1.0,
                           trigLevel=0.5,
                           triggerDelay=0,
                           triggerTimeout=0,
                           bufferCount=512)

################################################################################
# 读出混频
################################################################################

async def modulation_read(measure,delta,tdelay=1100,repeats=512,phase=0.0,amp=0.3,rname=['Readout_I','Readout_Q']):
    t_list, ats, awg = measure.t_list, measure.ats, measure.awg['awgread']
    await awg.stop()
    twidth, n, measure.readlen = tdelay, len(delta), tdelay
    wavelen = (twidth // 64) * 64
    if wavelen < twidth:
        wavelen += 64
    measure.wavelen = int(wavelen) 

    await genwaveform(awg,rname,[1,5])
    
    ringup = 100
    # pulse1 = wn.square((wavelen-ringup)/1e9) >> ((wavelen+ringup)/ 1e9 / 2)
    pulse1 = wn.square(wavelen/1e9) >> (wavelen/ 1e9 / 2 + ringup / 1e9)
    pulse2 = wn.square(ringup/1e9)>>(ringup/2/1e9)
    pulse = amp*pulse1+pulse2

    mrkp1 = wn.square(25000e-9) << (25000e-9 / 2 + 300e-9)
    mrkp2 = wn.square(wavelen/1e9) >> (wavelen / 1e9 / 2 + 440 / 1e9 + ringup / 1e9)
    # mrkp3 = wn.square(4000/1e9) >> 1000/1e9
    mrkp4 = wn.square(5000/1e9) << (2500+84995)/1e9
    I, Q = wn.zero(), wn.zero()
    for i in delta:
        wav_I, wav_Q = wn.mixing(pulse,phase=phase,freq=i,ratioIQ=-1.0)
        I, Q = I + wav_I, Q + wav_Q
    pulselist = I, Q, (mrkp2,), (mrkp4,mrkp4,mrkp1,mrkp4)
    await writeWave(awg,rname,pulselist,True,mark=True)
    await awg.setValue('Vpp',1.5*n,ch=1)
    await awg.setValue('Vpp',1.5*n,ch=5)
    await awg.write('*WAI')
    await ats_setup(ats,delta,l=tdelay,repeats=repeats)
    await couldRun(awg)
    return pulselist

################################################################################
# 激励混频
################################################################################

async def modulation_ex(qubit,measure,w=20000e-9,delta_ex=[0],shift=0):
    t_list, awg = measure.t_list, measure.awg['awg132']
    wf, qname, n = WF(t_list), ['awg132_ch7','awg132_ch8'], len(delta_ex)
    await genwaveform(awg,qname,[7,8])
    pulse = wn.square(w) << (w / 2 + 200e-9+shift)
    await writeWave(awg,qname,(pulse,pulse,pulse,pulse),mark=True)
    await couldRun(awg)

################################################################################s
# Rabi波形
################################################################################

async def rabiWave(envelopename='square',nwave=1,amp=1,during=75e-9,shift=0,Delta_lo=110e6,phase=0,phaseDiff=0,DRAGScaling=None,single=False):
    shift += 200e-9
    envelope = whichEnvelope(envelopename)
    if single:
        wave = ((envelope(2*during) << (shift+during))) *amp
    else:
        wave = (((envelope(during) << (shift+during/2))) + ((envelope(during) << (shift+during/2*3)))) * amp
    
    # mwav = (wn.square(2*during+380e-9) << (during+190e-9+10e-9)) * amp
    mwav = wn.square(2*during) << (during+shift)
    pulse, mpulse = wn.zero(), wn.zero()
    for i in range(nwave):
        pulse += (wave << 2*i*during)
        mpulse += (mwav << 2*i*during)
    wav_I, wav_Q = wn.mixing(pulse,phase=phase,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)

    return wav_I, wav_Q, mpulse, mpulse

async def Rabi_sequence(measure,kind,awg,t_rabi,amp=1,nwave=1,delta_lo=110e6,envelopename='hanning'):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    measure.envelopename, measure.amp = envelopename, amp
    await awg.stop()
    for j,i in enumerate(tqdm(t_rabi,desc=''.join(('Rabi_',kind)))):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await rabiWave(envelopename=envelopename,nwave=nwave,amp=amp,during=i/1e9,\
            Delta_lo=delta_lo,phase=0,phaseDiff=0,DRAGScaling=None)
        await writeWave(awg,name_ch,pulse,mark=False)

async def Rabipower_sequence(measure,kind,awg,t_rabi,amp=1,nwave=1,delta_lo=110e6,envelopename='hanning'):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    measure.envelopename = envelopename
    await awg.stop()
    for j,i in enumerate(tqdm(amp,desc=''.join(('Rabipower_',kind)))):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await rabiWave(envelopename=envelopename,nwave=nwave,amp=i,during=t_rabi/1e9,\
            Delta_lo=delta_lo,phase=0,phaseDiff=0,DRAGScaling=None,single=True)
        await writeWave(awg,name_ch,pulse)

################################################################################
# pipulseDetune波形
################################################################################

async def pipulseDetunewave(measure,awg,pilen,n,alpha,delta,name_ch,phaseDiff=0):
    shift = 200/1e9
    pilen = pilen / 1e9
    envelope = whichEnvelope(measure.envelopename)
    I, Q, mrk1, mrk2 = wn.zero(), wn.zero(), wn.zero(), wn.zero()
    for i in [np.pi,0]*n:
        pulse = (envelope(pilen) << (0.5*pilen+shift)) + (envelope(pilen) << (1.5*pilen + shift))
        shift += 2*pilen
        wav_I, wav_Q = wn.mixing(pulse,phase=i,freq=delta,phaseDiff=phaseDiff,ratioIQ=-1.0,DRAGScaling=alpha)
        I, Q = I+wav_I, Q+wav_Q
    await writeWave(awg,name_ch,(I,Q,mrk1,mrk2))
    return I,Q,mrk1,mrk2
    
################################################################################
# T1波形
################################################################################

async def T1_sequence(measure,kind,awg,t_rabi,pi_len,shift=0,Delta_lo=110e6):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await awg.stop()
    for j,i in enumerate(tqdm(t_rabi,desc='T1_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await rabiWave(envelopename=measure.envelopename,amp=measure.amp,during=pi_len/1e9,shift=(i+shift)*1e-9,\
            Delta_lo=Delta_lo,phaseDiff=0,DRAGScaling=None)
        await writeWave(awg,name_ch,pulse)

################################################################################
# Ramsey及SpinEcho,CPMG, PDD波形
################################################################################

async def coherenceWave(measure,t_run=0,during=75e-9,n_wave=0,seqtype='CPMG',detune=3e6,shift=200e-9,Delta_lo=110e6):
    amp = measure.amp
    envelopename = measure.envelopename
    envelope = whichEnvelope(envelopename)
    pulse1 = envelope(during) << (during/2+shift)
    wavI1, wavQ1 = wn.mixing(pulse1,phase=2*np.pi*detune*t_run,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    pulse3 = envelope(during) << (t_run+(2*n_wave+1.5)*during+shift)
    wavI3, wavQ3 = wn.mixing(pulse3,phase=0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)

    if seqtype == 'CPMG':
        pulse2, step = wn.zero(), t_run / n_wave
        for i in range(n_wave):
            pulse2 += envelope(2*during) << ((i+0.5)*step+(i+1)*2*during+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    if seqtype == 'PDD':
        pulse2, step = wn.zero(), t_run / (n_wave + 1)
        for i in range(n_wave):
            pulse2 += envelope(2*during) << ((i+1)*step+(i+1)*2*during+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wavI, wavQ = wavI1 + wavI2 + wavI3, wavQ1 + wavQ2 + wavQ3
    
    return wavI*amp, wavQ*amp, wn.zero(), wn.zero()

async def Coherence_sequence(measure,kind,awg,t_rabi,halfpi,n_wave=0,seqtype='PDD'):
    # t_range, sample_rate = measure.t_range, measure.sample_rate

    for j,i in enumerate(tqdm(t_rabi,desc='Coherence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await coherenceWave(measure,i/1e9,halfpi/1e9,n_wave,seqtype)
        await writeWave(awg,name_ch,pulse)

################################################################################
# Z_cross波形
################################################################################

async def Z_cross_sequence(measure,kind_z,kind_ex,awg_z,awg_ex,v_rabi,halfpi):
    #t_range, sample_rate = measure.t_range, measure.sample_rate
    for j,i in enumerate(tqdm(v_rabi,desc='Z_cross_sequence')):
        name_ch = [measure.wave[kind_ex][0][j],measure.wave[kind_ex][1][j]]
        pulse_ex = await coherenceWave(measure.envelopename,300/1e9,halfpi/1e9,0,'PDD')
        await writeWave(awg_ex,name_ch,pulse_ex)
        pulse_z = (wn.square(200/1e9) << (50+halfpi+200)/1e9) * i
        await writeWave(awg_z,np.array(measure.wave[kind_z])[:,j],pulse_z)

################################################################################
# Speccrosstalk波形
################################################################################

async def Speccrosstalk_sequence(measure,kind_z,kind_ex,awg_z,awg_ex,v_rabi,halfpi):
    #t_range, sample_rate = measure.t_range, measure.sample_rate
    for j,i in enumerate(tqdm(v_rabi,desc='Speccrosstalk_sequence')):
        name_ch = [measure.wave[kind_ex][0][j],measure.wave[kind_ex][1][j]]
        pulse_ex = await rabiWave(envelopename=measure.envelopename,during=halfpi/1e9,shift=(100)*1e-9,phaseDiff=0,DRAGScaling=None)
        await writeWave(awg_ex,name_ch,pulse_ex)
        pulse_z = (wn.square(1100/1e9) << (200)/1e9) * i
        await writeWave(awg_z,np.array(measure.wave[kind_z])[:,j],pulse_z)

################################################################################
# AC-Stark波形
################################################################################

async def ac_stark_wave(measure):
    awg = measure.awg['awgread']
    pulse_read = await modulation_read(measure,measure.delta,tdelay=measure.readlen)
    width = 500e-9
    pulse = wn.square(width) << (width/2+3000e-9)
    I, Q = wn.zero(), wn.zero()
    for i in measure.delta:
        wav_I, wav_Q = wn.mixing(pulse,phase=0.0,freq=i,ratioIQ=-1.0)
        I, Q = I + wav_I, Q + wav_Q
    pulse_acstark = (I,Q,wn.zero(),wn.zero())
    pulselist = np.array(pulse_read) + np.array(pulse_acstark)
    await writeWave(awg,['Readout_I','Readout_Q'],pulselist)
    
################################################################################
# ZPulse波形
################################################################################

async def zWave(awg,name,volt=0.4,during=500e-9,shift=1000e-9,offset=0):

    pulse = (wn.square(during) << (during/2 + shift)) * volt + offset
    pulselist = (pulse,) 
    await writeWave(awg,name,pulselist)
    return pulselist

################################################################################
# 偏置sequence
################################################################################

async def Z_sequence(measure,kind,z_awg,t_rabi,volt,shift=0):
    # t_range, sample_rate = measure.t_range, measure.sample_rate
    await z_awg.stop()
    for j,i in enumerate(tqdm(t_rabi,desc='Z_sequence')):
        name_ch = [measure.wave[kind][0][j]]
        pulse = await zWave(z_awg,name_ch,volt=volt,during=i*1e-9,shift=shift)
        await writeWave(z_awg,name_ch,pulse)

################################################################################
# RB波形
################################################################################

def genXY(during,shift=0e-9,tdelay=0e-9,pulse='pi',envelopename='square'):
    shift += 200e-9
    envelope = whichEnvelope(envelopename)
    if pulse == 'halfpi':
        pulse = envelope(during) << (shift+during/2)
    if pulse == 'pi':
        pulse = (envelope(during) << (shift+during/2)) + (envelope(during) << (shift+during/2*3))
    
    return pulse

def genParas(x):
    if x == 'I':
        paras = (0,'pi')
    elif x == 'X':
        paras = (0,'pi')
    elif x == 'Xhalf':
        paras = (0,'halfpi')
    elif x == 'Xnhalf':
        paras = (np.pi,'halfpi')
    elif x == 'Y':
        paras = (np.pi/2,'pi')
    elif x == 'Yhalf':
        paras = (np.pi/2,'halfpi')
    elif x == 'Ynhalf':
        paras = (3*np.pi/2,'halfpi')
    return paras

async def rbWave(measure,m,gate,pilen,Delta_lo=110e6,shift=0,phaseDiff=0.0,DRAGScaling=None):
    op = {'1':['I'],'2':['X'],'3':['Xhalf'],'4':['Xnhalf'],'5':['Y'],'6':['Yhalf'],'7':['Ynhalf'],
        '8':['X','Y'],'9':['Xhalf','Yhalf','Xnhalf'],'10':['Xhalf','Ynhalf','Xnhalf'],'11':['Ynhalf','X'],
        '12':['Yhalf','X'],'13':['Xhalf','Y'],'14':['Xnhalf','Y'],'15':['Xhalf','Yhalf','Xhalf'],'16':['Xnhalf','Yhalf','Xnhalf'],
        '17':['Xhalf','Yhalf'],'18':['Xnhalf','Yhalf'],'19':['Xhalf','Ynhalf'],'20':['Xnhalf','Ynhalf'],
        '21':['Ynhalf','Xnhalf'],'22':['Ynhalf','Xhalf'],'23':['Yhalf','Xnhalf'],'24':['Yhalf','Xhalf']}

    mseq = mx.cliffordGroup_single(m,gate)
    if mseq == []:
        return
    rotseq = []
    for i in mseq[::-1]:
        rotseq += op[i]
    waveseq_I, waveseq_Q, wav = wn.zero(), wn.zero(), wn.zero()
    # rotseq = ['Xhalf','Xnhalf','Yhalf','Ynhalf']*m
    # if rotseq == []:
    #     return
    # print(rotseq)
    for i in rotseq:
        paras = genParas(i)
        if i == 'I':
            waveseq_I += wn.zero()
            waveseq_Q += wn.zero()
            # continue
        else:
            pulse = genXY(during=pilen,pulse=paras[1],envelopename=measure.envelopename)
            cosPulse = pulse << shift
            phi = paras[0]
            wav_I, wav_Q = wn.mixing(cosPulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
            waveseq_I += wav_I
            waveseq_Q += wav_Q
            
        if paras[1] == 'pi':
            shift += 2*pilen
        if paras[1] == 'halfpi':
            shift += pilen

    return waveseq_I, waveseq_Q, wn.zero(), wn.zero()
    
async def rb_sequence(measure,awg,kind,m,gate,pilen):
    await awg.stop()
    for j,i in enumerate(tqdm(range(len(measure.wave[kind][0])),desc='RB_sequence')):
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await rbWave(measure,m,gate,pilen*1e-9)
        await writeWave(awg,name_ch,pulse)
    await awg.query('*OPC?')

################################################################################
# 单比特tomo波形
################################################################################

async def tomoWave(envelopename='square',during=0,shift=0,Delta_lo=110e6,axis='X',DRAGScaling=None,phaseDiff=0):
    shift += 200e-9
    envelope = whichEnvelope(envelopename)
    if axis == 'X':
        phi = 0
        pulse = (((envelope(during) << (shift+during/2))) + ((envelope(during) << (shift+during/2*3))))
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Xhalf':
        phi = 0
        pulse = envelope(during) << (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Xnhalf':
        phi = np.pi
        pulse = envelope(during) << (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Y':
        phi = np.pi/2
        pulse = (((envelope(during) << (shift+during/2))) + ((envelope(during) << (shift+during/2*3))))
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Ynhalf':
        phi = -np.pi/2
        pulse = envelope(during) << (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Yhalf':
        phi = np.pi/2
        pulse = envelope(during) << (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Z':
        wav_I, wav_Q = wn.zero(), wn.zero()
    return wav_I, wav_Q, wn.zero(), wn.zero()

################################################################################
# 单比特tomo测试
################################################################################

async def tomoTest(measure,awg,t,halfpi,axis,name,DRAGScaling):
    gatepulse = await rabiWave(measure.envelopename,during=t/1e9,shift=(halfpi)/1e9,\
        Delta_lo=110e6,DRAGScaling=DRAGScaling)
    # gatepulse = await rabiWave(envelopename=measure.envelopename,during=halfpi/1e9)
    tomopulse = await tomoWave(measure.envelopename,halfpi/1e9,\
        Delta_lo=110e6,axis=axis,DRAGScaling=DRAGScaling)
    pulse = np.array(gatepulse) + np.array(tomopulse)
    await writeWave(awg,name,pulse)

################################################################################
# ramseyZpulse波形
################################################################################

async def ramseyZwave(measure,awg,halfpi,axis,name,DRAGScaling,shift):
    t_int = 100e-9
    gatepulse = await tomoWave(measure.envelopename,halfpi/1e9,shift=(t_int+(shift+halfpi)/1e9),\
        Delta_lo=110e6,axis='Xhalf',DRAGScaling=DRAGScaling)
    # pulsespinecho = await rabiWave(measure.envelopename,during=halfpi/1e9,shift=(t_int+halfpi/1e9),\
    #     phase=-np.pi/2,Delta_lo=110e6,DRAGScaling=DRAGScaling)
    tomopulse = await tomoWave(measure.envelopename,halfpi/1e9,shift=shift/1e9,\
        Delta_lo=110e6,axis=axis,DRAGScaling=DRAGScaling)
    pulse = np.array(gatepulse) + np.array(tomopulse)
    await writeWave(awg,name,pulse)

################################################################################
# ramseyZpulse_chen波形
################################################################################

async def ramseyZwave_chen(measure,awg,halfpi,axis,shift,name,DRAGScaling):
    t_int = 700e-9
    gatepulse = await tomoWave(measure.envelopename,halfpi/1e9,shift=(t_int+shift+halfpi/1e9),\
        Delta_lo=110e6,axis='Xhalf',DRAGScaling=DRAGScaling)
    # pulsespinecho = await rabiWave(measure.envelopename,during=halfpi/1e9,shift=(t_int+halfpi/1e9),\
    #     phase=-np.pi/2,Delta_lo=110e6,DRAGScaling=DRAGScaling)
    tomopulse = await tomoWave(measure.envelopename,halfpi/1e9,Delta_lo=110e6,axis=axis,DRAGScaling=DRAGScaling)
    pulse = np.array(gatepulse) + np.array(tomopulse)
    await writeWave(awg,name,pulse)

################################################################################
# AllXY drag detune
################################################################################

async def dragDetunewave(measure,awg,pilen,coef,axis,name_ch):
    pulse1 = await tomoWave(measure.envelopename,during=pilen/1e9,Delta_lo=110e6,axis=axis[0],DRAGScaling=coef)
    pulse2 = await tomoWave(measure.envelopename,during=pilen/1e9,Delta_lo=110e6,axis=axis[1],DRAGScaling=coef)
    pulse = np.array(pulse1) + np.array(pulse2)
    await writeWave(awg,name_ch,pulse)
    
################################################################################
# dragcoefHD
################################################################################

async def HDWave(measure,awg,pilen,coef,axis,nwave,name_ch):
    pulseequator2 = await tomoWave(measure.envelopename,during=pilen/1e9,Delta_lo=110e6,axis='Xhalf',DRAGScaling=coef,phaseDiff=0)
    pulsem, shift = np.array([wn.zero()]*4), 0
    for i in range(nwave):
        shift += pilen/1e9 +10e-9
        pulse1 = await tomoWave(measure.envelopename,during=pilen/1e9,shift=shift,Delta_lo=110e6,axis=axis[0],DRAGScaling=coef,phaseDiff=0)
        shift += pilen/1e9 + 10e-9
        pulse2 = await tomoWave(measure.envelopename,during=pilen/1e9,shift=shift,Delta_lo=110e6,axis=axis[1],DRAGScaling=coef,phaseDiff=0)
        pulsem += (np.array(pulse1)+np.array(pulse2))
    pulseequator1 = await tomoWave(measure.envelopename,during=pilen/1e9,shift=(shift+pilen/1e9+10e-9),\
        Delta_lo=110e6,axis='Xhalf',DRAGScaling=coef,phaseDiff=0)
    pulse = np.array(pulseequator1) + np.array(pulseequator2) + np.array(pulsem)
    await writeWave(awg,name_ch,pulse)

################################################################################
# RTO
################################################################################

async def rtoWave(measure,awg,pilen,name_ch):
    envelopename = measure.envelopename
    pulsegate = await tomoWave(envelopename,during=pilen/1e9,Delta_lo=110e6,axis='Xhalf',shift=(pilen+100)/1e9,DRAGScaling=None)
    pulsetomoy = await tomoWave(envelopename,during=pilen/1e9,Delta_lo=110e6,axis='Ynhalf',DRAGScaling=None)
    pulsetomox = await tomoWave(envelopename,during=pilen/1e9,Delta_lo=110e6,axis='Xhalf',DRAGScaling=None)
    pulse1 = np.array(pulsegate)+np.array(pulsetomoy)
    pulse2 = np.array(pulsegate)+np.array(pulsetomox)
    await writeWave(awg,name_ch[0],pulse1)
    await writeWave(awg,name_ch[1],pulse2)