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
from qulab import computewave as cw

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
        self.att = {}
        self.inststate = 0
        self.t_list = np.linspace(0,100000,250000)
        self.t_range = (-90e-6,10e-6)
        self.sample_rate = 2.5e9
        
class qubitCollections():
    def __init__(self,qubits,q_target=None):
        self.qubits = {i.q_name:i for i in qubits}
        if q_target != None:
            qasdict = self.qubits[q_target]._asdict()
            for i in qasdict:
                # if not hasattr(self,i):
                self.__setattr__(i,qasdict[i])
        else:
            self.inst = {i.q_name:i.inst for i in qubits}
            self.q_name = 'allqubits'

        self.f_lo = np.array([i.f_lo[0] for i in qubits])
        self.f_ex = np.array([i.f_ex[0] for i in qubits])

    def qubitExecute(self,q_target=[]):
        q = []
        for i in q_target:
            q.append(self.qubits[i])
        return q

################################################################################
### 设置衰减器
################################################################################

class Att_Setup():
    
    def __init__(self,measure,com):
        self.com = com
        try:
            ser = serial.Serial(self.com,baudrate=115200, parity='N',bytesize=8, stopbits=1, timeout=1)
            self.ser = ser
        except:
            self.ser = measure.att[com]
        measure.att[com] = self.ser
        if self.ser.isOpen():    # make sure port is open     
            print(self.ser.name + ' open...')
            # self.ser.write(b'*IDN?\n')
            # x = self.ser.readline().decode().split('\r''\n')
            # print(x[0])
            self.ser.write(b'ATT?\n')
            y = self.ser.readline().decode().split('\r''\n')
            print('last ATT',y[0])

    def close(self):
        self.ser.close()

    def Att(self,att,closeinst=True):
        self.ser.write(b'ATT %f\n'%att)
        time.sleep(1)
        self.ser.write(b'ATT?\n')
        y = self.ser.readline().decode().split('\r''\n')
        print('now ATT',y[0])

        if closeinst:
            self.close()
   
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

async def modulation_read(measure,delta,tdelay=1100,repeats=512,rname=['Readout_I','Readout_Q']):
    t_list, ats, awg = measure.t_list, measure.ats, measure.awg['awgread']
    await awg.stop()
    twidth, n, measure.readlen = tdelay, len(delta), tdelay
    wavelen = (twidth // 64) * 64
    if wavelen < twidth:
        wavelen += 64
    # f, phi, t_end, width, height = [delta,delta], [[0]*n,[0]*n], \
    #     [[90000+0.1*wavelen],[90000+wavelen]], [[0.1*wavelen],[0.9*wavelen]], [[1],[0.5]]
    f, phi, t_end, width, height = [delta], [[0]*n], [[90000+wavelen]], [[wavelen]], [[1]]
    wf = WF(t_list)
    measure.wavelen = int(wavelen) 
    await awg.create_waveform(name=rname[1], length=len(t_list), format=None)
    await awg.create_waveform(name=rname[0], length=len(t_list), format=None)
    sample = wf.square_envelope(f,phi,t_end,width,height)
    await awg.update_waveform(sample/np.abs(sample).max(),name=rname[1])
    
    # f, phi, t_end, width, height = [delta,delta], [[np.pi/2]*n,[np.pi/2]*n], \
    #     [[90000+0.1*wavelen],[90000+wavelen]], [[0.1*wavelen],[0.9*wavelen]], [[1],[0.5]]
    f, phi, t_end, width, height = [delta], [[np.pi/2]*n], [[90000+wavelen]], [[wavelen]], [[1]]
    sample = wf.square_envelope(f,phi,t_end,width,height)
    await awg.update_waveform(sample/np.abs(sample).max(),name=rname[0])
    
    # sample = wf.square_wave(t_end=[90345+128/2+wavelen],width=[wavelen],height=[1])
    sample = wf.square_wave(t_end=[90405+wavelen],width=[wavelen],height=[1])
    await awg.update_marker(name=rname[1],mk1=sample)
    sample = wf.square_wave(t_end=[90030],width=[25050],height=[1])
    await awg.update_marker(name=rname[0],mk1=sample)
    await ats_setup(ats,delta,l=tdelay,repeats=repeats)

    await awg.use_waveform(name=rname[0],ch=7)
    await awg.use_waveform(name=rname[1],ch=8)
    await awg.setValue('Vpp',1*n,ch=7)
    await awg.setValue('Vpp',1*n,ch=8)
    await awg.write('*WAI')

    await awg.output_on(ch=7)
    await awg.output_on(ch=8)
    await awg.run()
    await awg.query('*OPC?')
    
################################################################################
### 激励混频
################################################################################

async def modulation_ex(qubit,measure,w=25000,delta_ex=[0]):
    t_list, awg = measure.t_list, qubit.inst['ex_awg']
    await measure.awg[awg].stop()
    wf, Iname, Qname, n = WF(t_list), qubit.q_name+'_I', qubit.q_name+'_Q', len(delta_ex)
    await measure.awg[awg].create_waveform(name=Qname, length=len(t_list), format=None)
    await measure.awg[awg].create_waveform(name=Iname, length=len(t_list), format=None)

    f,phi,height,t_end,width = delta_ex,[0]*n,[1],[89700],[w]
    sample = wf.square_wave(t_end,width,height)
    await measure.awg[awg].update_waveform(sample,name=Qname)
    await measure.awg[awg].update_marker(name=Qname,mk1=sample)
    f,phi,height,t_end,width = delta_ex,[np.pi/2]*n,[1],[89700],[w]
    sample = wf.square_wave(t_end,width,height)
    await measure.awg[awg].update_waveform(sample,name=Iname)

    await measure.awg[awg].use_waveform(name=Iname,ch=qubit.inst['ex_ch'][0])
    await measure.awg[awg].use_waveform(name=Qname,ch=qubit.inst['ex_ch'][1])
    await measure.awg[awg].write('*WAI')

    await measure.awg[awg].output_on(ch=qubit.inst['ex_ch'][0])
    await measure.awg[awg].output_on(ch=qubit.inst['ex_ch'][1])
    await measure.awg[awg].run()
    await measure.awg[awg].query('*OPC?')

################################################################################
### AWG同步
################################################################################

async def awgSync(measure):
    t_list = measure.t_list
    wf = WF(t_list)
    for i in measure.awg:
        await measure.awg[i].create_waveform(name=''.join((i,'_sync1')),length=len(measure.t_list),format=None)
        await measure.awg[i].create_sequence(name=''.join((i,'_syncSeq')),steps=3,tracks=1)
        height,t_end,width = [1],[50000],[2000]
        sample = wf.square_wave(t_end,width,height)
        await measure.awg[i].update_waveform(sample,name=''.join((i,'_sync1')))
        for j in range(3):
            j += 1
            goto = 'FIRST' if j == 3 else 'NEXT'
            await measure.awg[i].set_sequence_step(name=''.join((i,'_syncSeq')),sub_name=[''.join((i,'_sync1'))],\
                                                   step=j,wait='ATR',goto=goto,repeat=1,jump=None)
        await measure.awg[i].use_sequence(name=''.join((i,'_syncSeq')),channels=[1])
        await measure.awg[i].query('*OPC?')
        await measure.awg[i].output_on(ch=1)
        await measure.awg[i].setValue('Force Jump',1,ch=1)
        await measure.awg[i].run()  

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
            if 'dc' in i:
                current = await inst[i].getValue('Offset')
                load = await inst[i].getValue('Load')
                load = eval((load).strip('\n'))  
                load = 'high Z' if load != 50 else 50
                err = (await inst[i].query('syst:err?')).strip('\n').split(',')
                sm = {'offset':current,'load':load,'error':err[0]}
                state[i] = sm
            if 'psg' in i:
                freq = await inst[i].getValue('Frequency')
                power = await inst[i].getValue('Power')
                Output = await inst[i].getValue('Output')
                Moutput = await inst[i].getValue('Moutput')
                Mform = await inst[i].getValue('Mform')
                err = (await inst[i].query('syst:err?')).strip('\n').split(',')
                sm = {'freq':'%fGHz'%(freq/1e9),'power':'%fdBm'%power,'output':Output,'moutput':Moutput,\
                    'mform':Mform,'error':err[0]}
                state[i] = sm
        finally:
            pass
    measure.inststate = state
    return state

################################################################################
### 初始化仪器
################################################################################

async def InitInst(measure,psgdc=True,awgch=True,clearwaveseq=None):
    if psgdc:
        await close(measure)
    if awgch:
        for i in measure.awg:
            for j in range(8):
                await measure.awg[i].output_off(ch=j+1)
    if clearwaveseq != None:
        for i in clearwaveseq:
            await measure.awg[i].stop()
            #await measure.awg[i].query('*OPC?')
            await measure.awg[i].clear_waveform_list()
            await measure.awg[i].clear_sequence_list()
            m = list(measure.wave.keys())
            for j in m:
                if i in j:
                    del measure.wave[j]
                if i == 'awg134':
                    del measure.wave['Read']


################################################################################
### 恢复仪器最近状态
################################################################################

async def RecoverInst(measure,state=None):
    if state is None:
        state = measure.inststate
    for i in state:
        if 'dc' in i:
            await measure.dc[i].DC(state[i]['offset'])
        if 'psg' in i:
            await measure.psg[i].setValue('Frequency',eval(state[i]['freq'].strip('GHz'))*1e9)
            await measure.psg[i].setValue('Power',eval(state[i]['power'].strip('dBm')))
            if state[i]['output'] == '1':
                await measure.psg[i].setValue('Output','ON')
            else:
                await measure.psg[i].setValue('Output','OFF')

################################################################################
### S21
################################################################################

async def S21(qubit,measure,modulation=False,freq=None):
    #await jpa_switch(measure,state='OFF')
    if freq == None:
        f_lo, delta, n = await resn(np.array(qubit.f_lo))
        freq = np.linspace(-2.5,2.5,126)*1e6 + f_lo
        if modulation:
            await modulation_read(measure,delta,tdelay=measure.readlen)
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.awg['awgread'].run()
    await measure.awg['awgread'].query('*OPC?')
    await measure.awg['awgread'].output_on(ch=7)
    await measure.awg['awgread'].output_on(ch=8)
    for i in freq:
        await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
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
        #await ats_setup(measure.ats,delta)
        await modulation_read(measure,delta,tdelay=measure.readlen)
        base = 0
        for i in range(25):
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            base += Am + 1j*Bm
        base /= 25
    measure.base, measure.n, measure.delta, measure.f_lo = base, n, delta, np.array([f_lo])
    return f_lo, delta, n, f_res, base

################################################################################
### S21vsFlux
################################################################################

async def S21vsFlux(qubit,measure,current,calimatrix,modulation=False):

    # for i in current:
    #     await measure.dc[qubit.inst['dc']].DC(i)
    #     job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
    #     f_s21, s_s21 = await job.done()
    #     n = np.shape(s_s21)[1]
    #     yield [i]*n, f_s21, s_s21
    for i in current:
        clist = np.mat(calimatrix).I * np.mat([0,i,0,0]).T
        for k, j in enumerate(clist,start=2):
            await measure.dc[qubit.inst['q%d'%k]['dc']].DC(j)
        job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21
################################################################################
### S21vsFlux_awgoffset
################################################################################

async def S21vsFlux_awgoffset(qubit,measure,current,calimatrix,modulation=False):
    # for i in current:
    #     await measure.awg[qubit.inst['z_awg']].create_waveform(name=''.join((qubit.q_name,'_z')), length=len(measure.t_list), format=None)
    #     await measure.awg[qubit.inst['z_awg']].use_waveform(name=''.join((qubit.q_name,'_z')),ch=qubit.inst['z_ch'])
    #     await measure.awg[qubit.inst['z_awg']].query('*OPC?')
    #     await measure.awg[qubit.inst['z_awg']].output_on(ch=qubit.inst['z_ch'])
    #     await measure.awg[qubit.inst['z_awg']].setValue('Offset',i,ch=qubit.inst['z_ch'])
    #     # await cw.awgDC(measure.awg[qubit.inst['z_awg']],''.join((qubit.q_name,'_z')),i)
    #     await measure.awg[qubit.inst['z_awg']].run()
    #     await measure.awg[qubit.inst['z_awg']].query('*OPC?')
    #     job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
    #     f_s21, s_s21 = await job.done()
    #     n = np.shape(s_s21)[1]
    #     yield [i]*n, f_s21, s_s21
    for i in current:
        clist = np.mat(calimatrix).I * np.mat([0,i,0,0]).T
        for k, j in enumerate(clist,start=2):
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].create_waveform(name=''.join((qubit.q_name,'_z%d'%k)), length=len(measure.t_list), format=None)
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].use_waveform(name=''.join((qubit.q_name,'_z%d'%k)),ch=qubit.inst['q%d'%k]['z_ch'])
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].query('*OPC?')
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].output_on(ch=qubit.inst['q%d'%k]['z_ch'])
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].setValue('Offset',j,ch=qubit.inst['q%d'%k]['z_ch'])
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].run()
            await measure.awg[qubit.inst['q%d'%k]['z_awg']].query('*OPC?')
        job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
### S21vsPower
################################################################################

async def S21vsPower(qubit,measure,att,com='com7',modulation=False):
    att_setup = Att_Setup(com)
    for i in att:
        att_setup.Att(i,closeinst=False)
        job = Job(S21, (qubit,measure,modulation),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21
    att_setup.close()

################################################################################
### SingleSpec
################################################################################

async def singlespec(qubit,measure,freq,modulation=False,readponit=True):
    if readponit:
        f_lo, delta, n, f_res,base = await again(qubit,measure,modulation)
    else:
        n, base = len(measure.delta), measure.base
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    
    for i in freq:
        await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base

################################################################################
### SingleSpec扫电压
################################################################################

async def specbias(qubit,measure,ftarget,bias,modulation=False):
    await measure.dc[qubit.inst['dc']].DC(round(np.mean(bias),3))
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',ftarget)
    f_lo, delta, n, f_res,base = await again(qubit,measure,modulation)
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    for i in bias:
        await measure.dc[qubit.inst['dc']].DC(i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base

################################################################################
### Spec2d
################################################################################

async def spec2d(qubit,measure,freq,calimatrix,init,modulation=False):
    # current = np.linspace(-qubit.T_bias[0]*0.3,qubit.T_bias[0]*0.3,31) + qubit.T_bias[1] 
    current = np.linspace(-1.759*0.3,1.759*0.3,31) + 0.224
    # current = np.linspace(-0.8,0.8,33)
    qubit.inst['ex_awg'] = 'awg132'
    qubit.inst['ex_ch'] = [7, 8]
    qubit.inst['ex_lo'] = 'psg_ex2'
    await modulation_ex(qubit,measure)
    for i in current:
        # await measure.dc[qubit.inst['dc']].DC(i)
        clist = np.mat(calimatrix).I * np.mat(init).T * i
        for k, j in enumerate(clist,start=2):
            await measure.dc[qubit.inst['q%d'%k]['dc']].DC(j)
        # clist = np.mat(calimatrix).I * np.mat(init).T * i
        # for k, j in enumerate(clist,start=2):
        #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].create_waveform(name=''.join((qubit.q_name,'_z%d'%k)), length=len(measure.t_list), format=None)
        #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].use_waveform(name=''.join((qubit.q_name,'_z%d'%k)),ch=qubit.inst['q%d'%k]['z_ch'])
        #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].query('*OPC?')
        #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].output_on(ch=qubit.inst['q%d'%k]['z_ch'])
        #     await cw.zWave(measure.awg[qubit.inst['q%d'%k]['z_awg']],[''.join((qubit.q_name,'_z%d'%k))],qubit.inst['q%d'%k]['z_ch'],j[0,0],5000e-9,200e-9)
        #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].run()
        #     await measure.awg[qubit.inst['q%d'%k]['z_awg']].query('*OPC?')
        job = Job(singlespec, (qubit,measure,freq,modulation),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

################################################################################
### Rabi
################################################################################

async def rabi(qubit,measure,t_rabi,len_data,nwave=1,comwave=False):
    t, name = t_rabi[1:len_data+1], ''.join((qubit.inst['ex_awg'],'coherence_rabi'))
    t = np.array([t]*measure.n).T
    await cw.create_wavelist(measure,name,(qubit.inst['ex_awg'],['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,measure.awg[qubit.inst['ex_awg']],name)
    if comwave:
        await cw.Rabi_sequence(measure,name,qubit.inst['ex_awg'],t_rabi,nwave)
    await measure.awg[qubit.inst['ex_awg']].use_sequence(name,channels=[qubit.inst['ex_ch'][0],qubit.inst['ex_ch'][1]])
    await measure.awg[qubit.inst['ex_awg']].query('*OPC?')
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][0])
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][1])
    await measure.awg[qubit.inst['ex_awg']].run()
    await measure.awg[qubit.inst['ex_awg']].query('*OPC?')

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
### 优化pi脉冲
################################################################################
    
async def pipulseOpt(qubit,measure,nwave,wavlen):
    pilen = qubit.pi_len
    t = np.linspace(0.5*pilen,1.5*pilen,wavlen)
    for i in range(nwave):
        await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=(len(t)-1),awg=1)
        job = Job(rabi, (qubit,measure,t,(len(t)-1),2*i+1,True), max=500,avg=True)
        t_r, s_r = await job.done()
        yield [2*i+1]*measure.n, t_r, s_r

################################################################################
### Rabi_power
################################################################################

async def rabiPower(qubit,measure,att):
    name, awg = [qubit.q_name+'_I', qubit.q_name+'_Q'], measure.awg[qubit.inst['ex_awg']]
    await awg.create_waveform(name=name[0], length=len(t_list), format=None)
    await awg.create_waveform(name=name[1], length=len(t_list), format=None)
    await cw.rabiWave(awg,during=25e-9,name=name)
    await awg.use_waveform(name=name[0],ch=qubit.inst['ex_ch'][0])
    await awg.use_waveform(name=name[1],ch=qubit.inst['ex_ch'][1])
    await awg.output_on(ch=qubit.inst['ex_ch'][0])
    await awg.output_on(ch=qubit.inst['ex_ch'][1])
    await modulation_read(measure,measure.delta,tdelay=measure.readlen)
    # att_setup = Att_Setup(qubit.inst['com'])
    await measure.psg['psg_lo'].setValue('Output','ON')
    n = len(measure.base)
    for i in att:
        # att_setup.Att(i,False)
        await measure.psg[qubit.inst['ex_lo']].setValue('Power',i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i]*n, s - measure.base
    # att_setup.close()

################################################################################
### IQ-Mixer优化
################################################################################

async def optIQMixer(qubit,measure,att):
    name, awg = [qubit.q_name+'_I', qubit.q_name+'_Q'], measure.awg[qubit.inst['ex_awg']]
    await awg.create_waveform(name=name[0], length=len(t_list), format=None)
    await awg.create_waveform(name=name[1], length=len(t_list), format=None)
    await cw.rabiWave(awg,during=25e-9,name=name)
    await awg.use_waveform(name=name[0],ch=qubit.inst['ex_ch'][0])
    await awg.use_waveform(name=name[1],ch=qubit.inst['ex_ch'][1])
    await awg.output_on(ch=qubit.inst['ex_ch'][0])
    await awg.output_on(ch=qubit.inst['ex_ch'][1])
    await modulation_read(measure,measure.delta,tdelay=measure.readlen)
    # att_setup = Att_Setup(qubit.inst['com'])
    await measure.psg['psg_lo'].setValue('Output','ON')
    n = len(measure.base)
    for i in att:
        # att_setup.Att(i,False)
        await measure.psg[qubit.inst['ex_lo']].setValue('Power',i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i]*n, s - measure.base
    # att_setup.close()

################################################################################
### tomo测试
################################################################################

async def singleTomo(qubit,measure):
    awg = measure.awg[qubit.inst['ex_awg']]
    Iname, Qname = qubit.q_name+'_I', qubit.q_name+'_Q'
    await awg.create_waveform(name=Qname, length=len(t_list), format=None)
    await awg.create_waveform(name=Iname, length=len(t_list), format=None)
    await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    population = []
    for i in ['Z','X','Y']:
        await cw.tomoTest(awg,qubit.pi_len/1e9,i,[Iname,Qname])
        await awg.query('*OPC?')
        ch_A, ch_B = await measure.ats.getIQ()
        s = ch_A + 1j*ch_B
        population.append([i,s])
    yield  population


################################################################################
### Z_Pulse（Z与Readout时序矫正)
################################################################################

async def zPulse(qubit,measure,t):
    awg = measure.awg[qubit.inst['ex_awg']]
    freq = np.linspace(-200,200,201)*1e6 + qubit.f_ex[0] + qubit.delta_ex[0]
    await modulation_read(measure,measure.delta,tdelay=measure.readlen)
    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    await measure.awg[qubit.inst['z_awg']].create_waveform(name=zname[0], length=len(t_list), format=None)  
    await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,0.4)
    Iname, Qname = qubit.q_name+'_I', qubit.q_name+'_Q'
    await awg.create_waveform(name=Qname, length=len(t_list), format=None)
    await awg.create_waveform(name=Iname, length=len(t_list), format=None)
    await awg.use_waveform(name=Iname,ch=qubit.inst['ex_ch'][0])
    await awg.use_waveform(name=Qname,ch=qubit.inst['ex_ch'][1])
    await awg.output_on(ch=qubit.inst['ex_ch'][0])
    await awg.output_on(ch=qubit.inst['ex_ch'][1])
    n = len(measure.delta)
    for i in t:
        await awg.stop()
        await cw.rabiWave(awg,qubit.pi_len/1e9,shift=i/1e9,name=[Iname,Qname])
        await awg.run()
        await awg.query('*OPC?')
        job = Job(singlespec,(qubit,measure,freq,False,False),max=len(freq),auto_save=False)
        f, s = await job.done()
        #n = np.shape(s)[1] 
        yield [i]*n, f-qubit.delta_ex, s

################################################################################
### Z_Pulse（Z与Readout时序矫正) population
################################################################################

async def zPulse_pop(qubit,measure,t):
    awg = measure.awg[qubit.inst['ex_awg']]
    freq = np.linspace(-200,200,201)*1e6 + qubit.f_ex[0] + qubit.delta_ex[0]
    await modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=500)
    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    await measure.awg[qubit.inst['z_awg']].create_waveform(name=zname[0], length=len(t_list), format=None)  
    await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,0.4)
    Iname, Qname = qubit.q_name+'_I', qubit.q_name+'_Q'
    await awg.create_waveform(name=Qname, length=len(t_list), format=None)
    await awg.create_waveform(name=Iname, length=len(t_list), format=None)
    await awg.use_waveform(name=Iname,ch=qubit.inst['ex_ch'][0])
    await awg.use_waveform(name=Qname,ch=qubit.inst['ex_ch'][1])
    await awg.output_on(ch=qubit.inst['ex_ch'][0])
    await awg.output_on(ch=qubit.inst['ex_ch'][1])
    n = len(measure.delta)
    for i in t:
        await awg.stop()
        await cw.rabiWave(awg,qubit.pi_len/1e9,shift=i/1e9,name=[Iname,Qname])
        await awg.run()
        await awg.query('*OPC?')
        # ch_A, ch_B = await measure.ats.getIQ()
        # s = ch_A + 1j*ch_B
        # s = np.mean(s)
        # #ss, which = s[:,0], 1
        # #d = list(zip(np.real(ss),np.imag(ss)))
        # #y = measure.predict(d)
        # #pop = list(y).count(which)/len(y)
        # yield [i]*n, [s]*n
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i]*n, s - measure.base

################################################################################
### AC-Stark（XY与Readout时序矫正)
################################################################################

async def acStark(qubit,measure,t_rabi):
    # t, name = t_rabi[1:len_data+1], ''.join((qubit.inst['ex_awg'],'coherence')) #name--kind
    # t = np.array([t]*measure.n).T
    # await cw.create_wavelist(measure,name,(qubit.inst['ex_awg'],['I','Q'],len(t_rabi),len(measure.t_list)))
    # await cw.genSeq(measure,measure.awg[qubit.inst['ex_awg']],name)
    # if comwave:
    #     await cw.ac_stark_sequence(measure,qubit.inst['ex_awg'],name,qubit.pi_len,t_rabi)
    # await measure.awg[qubit.inst['ex_awg']].use_sequence(name,channels=[qubit.inst['ex_ch'][0],qubit.inst['ex_ch'][1]])
    # await measure.awg[qubit.inst['ex_awg']].query('*OPC?')
    # await cw.readSeq(measure,measure.awg['awgread'],'Read')
    # await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][0])
    # await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][1])
    # await measure.awg[qubit.inst['ex_awg']].run()
    # await measure.awg[qubit.inst['ex_awg']].query('*OPC?')

    # await measure.psg['psg_lo'].setValue('Output','ON')
    # await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    awg = measure.awg[qubit.inst['ex_awg']]
    freq = np.linspace(-60,60,121)*1e6 + qubit.f_ex[0] + qubit.delta_ex[0]
    Iname, Qname = qubit.q_name+'_I', qubit.q_name+'_Q'
    await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=500)
    await awg.create_waveform(name=Qname, length=len(t_list), format=None)
    await awg.create_waveform(name=Iname, length=len(t_list), format=None)
    await awg.use_waveform(name=Iname,ch=qubit.inst['ex_ch'][0])
    await awg.use_waveform(name=Qname,ch=qubit.inst['ex_ch'][1])
    await cw.ac_stark_sequence(measure,qubit.inst['ex_awg'],'',qubit.pi_len,0)
    await measure.awg['awgread'].query('*OPC?')
    for i in t_rabi:
        await awg.stop()
        await cw.rabiWave(awg,qubit.pi_len/1e9,shift=i/1e9,name=[Iname,Qname])
        await awg.run()
        await awg.query('*OPC?')
        job = Job(singlespec,(qubit,measure,freq,False,False),max=len(freq),auto_save=False)
        f, s = await job.done()
        #n = np.shape(s)[1]
        n = len(measure.delta)
        yield [i]*n, f-qubit.delta_ex, s
    
################################################################################
### 优化读出点
################################################################################

async def readOp(qubit,measure,modulation=False,freq=None):
    pilen = qubit.pi_len/1e9
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await awg.create_waveform(name=qname[0], length=len(t_list), format=None)
    await awg.create_waveform(name=qname[1], length=len(t_list), format=None)
    await cw.rabiWave(awg,during=pilen,name=qname)
    await awg.use_waveform(name=qname[0],ch=qubit.inst['ex_ch'][0])
    await awg.use_waveform(name=qname[1],ch=qubit.inst['ex_ch'][1])
    await awg.query('*OPC?')
    await awg.output_on(ch=qubit.inst['ex_ch'][0])
    await awg.output_on(ch=qubit.inst['ex_ch'][1])
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        job = Job(S21, (qubit,measure,modulation),auto_save=False,max=126)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [j]*n, f_s21, s_s21

################################################################################
### 优化读出长度
################################################################################

async def readWavelen(qubit,measure):
    pilen, t = qubit.pi_len/1e9, np.linspace(900,2000,21,dtype=np.int64)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await awg.create_waveform(name=qname[0], length=len(t_list), format=None)
    await awg.create_waveform(name=qname[1], length=len(t_list), format=None)
    await cw.rabiWave(awg,during=pilen,name=qname)
    await awg.use_waveform(name=qname[0],ch=qubit.inst['ex_ch'][0])
    await awg.use_waveform(name=qname[1],ch=qubit.inst['ex_ch'][1])
    await awg.query('*OPC?')
    await awg.output_on(ch=qubit.inst['ex_ch'][0])
    await awg.output_on(ch=qubit.inst['ex_ch'][1])
    for k in t:
        await modulation_read(measure,measure.delta,tdelay=int(k),repeats=5000)
        state = []
        for j, i in enumerate(['OFF','ON']):
            await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
            ch_A, ch_B = await measure.ats.getIQ()
            s = ch_A + 1j*ch_B
            state.append((j,np.mean(s),np.std(s)))
        yield k, state

################################################################################
### 临界判断
################################################################################

async def threshHold(qubit,measure,modulation=True):
    if modulation:
        pilen = qubit.pi_len/1e9
        await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
        awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
        await awg.create_waveform(name=qname[0], length=len(t_list), format=None)
        await awg.create_waveform(name=qname[1], length=len(t_list), format=None)
        await cw.rabiWave(awg,during=pilen,name=qname)
        await awg.use_waveform(name=qname[0],ch=qubit.inst['ex_ch'][0])
        await awg.use_waveform(name=qname[1],ch=qubit.inst['ex_ch'][1])
        await awg.query('*OPC?')
        await awg.output_on(ch=qubit.inst['ex_ch'][0])
        await awg.output_on(ch=qubit.inst['ex_ch'][1])
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        ch_A, ch_B = await measure.ats.getIQ()
        s = ch_A + 1j*ch_B
        yield j, s

################################################################################
### 优化读出振幅
################################################################################

async def readpowerOpt(qubit,measure,com,att):
    att_setup = Att_Setup(com)
    n = len(measure.delta)
    await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in att:
        att_setup.Att(i,False)
        job = Job(threshHold,(qubit,measure,False),auto_save=False,no_bar=True)
        st, s_st = await job.done()
        x, y = s_st[0][:,0], s_st[1][:,0]
        offmean, onmean = np.mean(x), np.mean(y)
        offstd, onstd = np.std(x), np.std(y)
        doff, don = np.abs(x-offmean), np.abs(y-onmean)
        popoff, popon = list(doff<offstd).count(True)/len(x), list(don<onstd).count(True)/len(y)
        yield [i]*n, [popoff]*n, [popon]*n
    att_setup.close()

################################################################################
### T1
################################################################################

async def T1(qubit,measure,t_rabi,len_data,comwave=False):
    t, name = t_rabi[1:len_data+1], ''.join((qubit.inst['ex_awg'],'coherence_rabi'))
    t = np.array([t]*measure.n).T
    await cw.create_wavelist(measure,name,(qubit.inst['ex_awg'],['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,measure.awg[qubit.inst['ex_awg']],name)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    if comwave:
        await cw.T1_sequence(measure,name,qubit.inst['ex_awg'],t_rabi,qubit.pi_len)
    await measure.awg[qubit.inst['ex_awg']].use_sequence(name,channels=[qubit.inst['ex_ch'][0],qubit.inst['ex_ch'][1]])
    await measure.awg[qubit.inst['ex_awg']].write('*WAI')
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][0])
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][1])
    await measure.awg[qubit.inst['ex_awg']].run()
    await measure.awg[qubit.inst['ex_awg']].query('*OPC?')

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
### Ramsey
################################################################################

async def Ramsey(qubit,measure,t_rabi,len_data,comwave=False):
    t, name = t_rabi[1:len_data+1], ''.join((qubit.inst['ex_awg'],'coherence'))
    t = np.array([t]*measure.n).T
    await cw.create_wavelist(measure,name,(qubit.inst['ex_awg'],['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,measure.awg[qubit.inst['ex_awg']],name)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    if comwave:
        await cw.Ramsey_sequence(measure,name,qubit.inst['ex_awg'],t_rabi,qubit.pi_len)
    await measure.awg[qubit.inst['ex_awg']].use_sequence(name,channels=[qubit.inst['ex_ch'][0],qubit.inst['ex_ch'][1]])
    await measure.awg[qubit.inst['ex_awg']].write('*WAI')
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][0])
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][1])
    await measure.awg[qubit.inst['ex_awg']].run()
    await measure.awg[qubit.inst['ex_awg']].query('*OPC?')

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
### SpinEcho
################################################################################

async def SpinEcho(qubit,measure,t_rabi,len_data,n_wave=0,seqtype='CPMG',comwave=False):
    t, name = t_rabi[1:len_data+1], ''.join((qubit.inst['ex_awg'],'coherence'))
    t = np.array([t]*measure.n).T
    await cw.create_wavelist(measure,name,(qubit.inst['ex_awg'],['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,measure.awg[qubit.inst['ex_awg']],name)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    if comwave:
        await cw.Coherence_sequence(measure,name,qubit.inst['ex_awg'],t_rabi,qubit.pi_len,n_wave,seqtype)
    await measure.awg[qubit.inst['ex_awg']].use_sequence(name,channels=[qubit.inst['ex_ch'][0],qubit.inst['ex_ch'][1]])
    await measure.awg[qubit.inst['ex_awg']].write('*WAI')
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][0])
    await measure.awg[qubit.inst['ex_awg']].output_on(ch=qubit.inst['ex_ch'][1])
    await measure.awg[qubit.inst['ex_awg']].run()
    await measure.awg[qubit.inst['ex_awg']].query('*OPC?')

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
### AWG crosstalk_cali
################################################################################

async def Z_cross(qubit_ex,qubit_z,measure,v_rabi,len_data,comwave=False):
    t, name_ex, name_z = v_rabi[1:len_data+1], ''.join((qubit_ex.inst['ex_awg'],'coherence')), ''.join((qubit_z.inst['z_awg'],'_z'))
    t = np.array([t]*measure.n).T
    await cw.create_wavelist(measure,name_z,(qubit_z.inst['z_awg'],['Z'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,measure.awg[qubit_z.inst['z_awg']],name_z)
    await cw.create_wavelist(measure,name_ex,(qubit_ex.inst['ex_awg'],['I','Q'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,measure.awg[qubit_ex.inst['ex_awg']],name_ex)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    if comwave:
        await cw.Z_cross_sequence(measure,name_z,name_ex,qubit_z.inst['z_awg'],qubit_ex.inst['ex_awg'],v_rabi,qubit_ex.pi_len)
    await measure.awg[qubit_ex.inst['ex_awg']].use_sequence(name_ex,channels=[qubit_ex.inst['ex_ch'][0],qubit_ex.inst['ex_ch'][1]])
    await measure.awg[qubit_ex.inst['ex_awg']].write('*WAI')
    await measure.awg[qubit_ex.inst['ex_awg']].output_on(ch=qubit_ex.inst['ex_ch'][0])
    await measure.awg[qubit_ex.inst['ex_awg']].output_on(ch=qubit_ex.inst['ex_ch'][1])
    await measure.awg[qubit_z.inst['z_awg']].use_sequence(name_z,channels=[qubit_z.inst['z_ch']])
    await measure.awg[qubit_z.inst['z_awg']].write('*WAI')
    await measure.awg[qubit_z.inst['z_awg']].output_on(ch=qubit_z.inst['z_ch'])
    await measure.awg[qubit_z.inst['z_awg']].run()
    await measure.awg[qubit_ex.inst['ex_awg']].run()
    await measure.awg[qubit_z.inst['z_awg']].query('*OPC?')
    await measure.awg[qubit_ex.inst['ex_awg']].query('*OPC?')

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await ats_setup(measure.ats,qubit_ex.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
### spec crosstalk cali
################################################################################

async def single_cs(measure,t,len_data):
    for i in range(500):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

async def crosstalkSpec(qubit_ex,qubit_z,measure,v_rabi,len_data,comwave):
    t, name_ex, name_z = v_rabi[1:len_data+1], ''.join((qubit_ex.inst['ex_awg'],'coherence')), ''.join((qubit_ex.inst['z_awg'],'_z'))
    t = np.array([t]*measure.n).T
    await cw.create_wavelist(measure,name_z,(qubit_ex.inst['z_awg'],['Z'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,measure.awg[qubit_ex.inst['z_awg']],name_z)
    await cw.create_wavelist(measure,name_ex,(qubit_ex.inst['ex_awg'],['I','Q'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,measure.awg[qubit_ex.inst['ex_awg']],name_ex)
    await cw.readSeq(measure,measure.awg['awgread'],'Read')
    await measure.awg[qubit_z.inst['z_awg']].create_waveform(name=''.join((qubit_z.q_name,'_z')), length=len(t_list), format=None)
    await measure.awg[qubit_z.inst['z_awg']].use_waveform(name=''.join((qubit_z.q_name,'_z')),ch=qubit_z.inst['z_ch'])
    await measure.awg[qubit_z.inst['z_awg']].output_on(ch=qubit_z.inst['z_ch'])
    if comwave:
        await cw.Speccrosstalk_sequence(measure,name_z,name_ex,qubit_ex.inst['z_awg'],qubit_ex.inst['ex_awg'],v_rabi,400)
    await measure.awg[qubit_ex.inst['ex_awg']].use_sequence(name_ex,channels=[qubit_ex.inst['ex_ch'][0],qubit_ex.inst['ex_ch'][1]])
    await measure.awg[qubit_ex.inst['ex_awg']].write('*WAI')
    await measure.awg[qubit_ex.inst['ex_awg']].output_on(ch=qubit_ex.inst['ex_ch'][0])
    await measure.awg[qubit_ex.inst['ex_awg']].output_on(ch=qubit_ex.inst['ex_ch'][1])
    await measure.awg[qubit_ex.inst['z_awg']].use_sequence(name_z,channels=[qubit_ex.inst['z_ch']])
    await measure.awg[qubit_ex.inst['z_awg']].write('*WAI')
    await measure.awg[qubit_ex.inst['z_awg']].output_on(ch=qubit_ex.inst['z_ch'])
    await measure.awg[qubit_ex.inst['z_awg']].run()
    await measure.awg[qubit_ex.inst['ex_awg']].run()
    await measure.awg[qubit_ex.inst['z_awg']].query('*OPC?')
    await measure.awg[qubit_ex.inst['ex_awg']].query('*OPC?')
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await ats_setup(measure.ats,qubit_ex.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in np.linspace(-1,1,11):
        await measure.awg[qubit_z.inst['z_awg']].stop()
        await cw.zWave(measure,measure.awg[qubit_z.inst['z_awg']],[''.join((qubit_z.q_name,'_z'))],qubit_z.inst['z_ch'],i,during=1100e-9,shift=200*1e-9)
        await measure.awg[qubit_z.inst['z_awg']].run()
        await measure.awg[qubit_z.inst['z_awg']].query('*OPC?')
        job = Job(single_cs,(measure,t,len_data),avg=True,max=500,auto_save=False,no_bar=True)
        v_bias, s =await job.done()
        n = np.shape(s)[1]
        yield [i]*n, v_bias, s

################################################################################
### Randomized Benchmarking
################################################################################

async def RB(qubit_ex,measure,mlist,len_data,comwave=False):

    name_ex = ''.join((qubit_ex.inst['ex_awg'],'rb'))
    for j in mlist:
        await cw.create_wavelist(measure,name_ex,(qubit_ex.inst['ex_awg'],['I','Q'],len_data,len(measure.t_list)))
        await cw.genSeq(measure,measure.awg[qubit_ex.inst['ex_awg']],name_ex)
        if comwave:
            await cw.rb_sequence(measure,qubit_ex.inst['ex_awg'],name_ex,j,qubit_ex.pi_len)
        await measure.awg[qubit_ex.inst['ex_awg']].use_sequence(name_ex,channels=[qubit_ex.inst['ex_ch'][0],qubit_ex.inst['ex_ch'][1]])
        await measure.awg[qubit_ex.inst['ex_awg']].query('*OPC?')
        await cw.readSeq(measure,measure.awg['awgread'],'Read')
        await measure.awg[qubit_ex.inst['ex_awg']].output_on(ch=qubit_ex.inst['ex_ch'][0])
        await measure.awg[qubit_ex.inst['ex_awg']].output_on(ch=qubit_ex.inst['ex_ch'][1])
        await measure.awg[qubit_ex.inst['ex_awg']].run()
        await measure.awg[qubit_ex.inst['ex_awg']].query('*OPC?')

        #await measure.psg['psg_lo'].setValue('Output','ON')
        await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
        await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
        s = []
        for i in range(3000):
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s.append((Am + 1j*Bm)[:,0])

        yield j, np.array(s)
    
      