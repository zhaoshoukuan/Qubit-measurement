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
import numpy as np, sympy as sy, serial, time
from easydl import clear_output
from qulab.job import Job
from qulab.wavepoint import WAVE_FORM as WF
from collections import Iterable
from tqdm import tqdm_notebook as tqdm
from qulab import computewave as cw, optimize as op

t_list = np.linspace(0,100000,250000)
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

class simpleclass():
    def __init__(self):
        pass

class common():
    def __init__(self,freqall,ats,dc,psg,awg,attinst,jpa,qubits=None):
        self.freqall = freqall
        self.ats = ats
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.jpa = jpa
        self.qubits = {i.q_name:i for i in qubits} if qubits is not None else {}
        self.wave = {}
        self.attinst = attinst
        self.com_lst = [f'com{i}' for i in np.arange(3,15)]
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
# 设置衰减器
################################################################################

class Att_Setup():
    
    def __init__(self,measure=None,com='com7'):
        if measure == None:
            measure = simpleclass()
            measure.att = {}
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
# 并行处理
################################################################################

def concurrence(task):
    loop = asyncio.get_event_loop()
    for i in task:
        loop.run_until_complete(asyncio.wait(i))
    loop.close()

################################################################################
# AWG恢复设置
################################################################################

async def resetAwg(awg):
    for i in awg:  
        await awg[i].setValue('Run Mode','Triggered')   # Triggered, Continuous
        if i == 'awg_trig':
            await cw.genwaveform(awg[i],['Readout_Q'],[5])
            await cw.couldRun(awg[i])
            # await awg[i].write('TRIGg:SOUR %s'%'INT')   
            await awg[i].write('TRIGGER:INTERVAL %f'%110e-6) 
            for m in range(8):
                await awg[i].write('SOUR%d:TINP %s'%((m+1),'ITR'))
        for j in range(8):
            await awg[i].setValue('Vpp',0.5,ch=j+1)
            await awg[i].setValue('Offset',0,ch=j+1)
            await awg[i].write('OUTPUT%d:WVALUE:ANALOG:STATE %s'%(j+1,'FIRST'))          #FIRST, ZERO
            for k in range(4):
                await awg[i].write('OUTPUT%d:WVALUE:MARKER%d %s'%(j+1,(k+1),'FIRST'))            #FIRST, LOW, HIGH

################################################################################
# AWG同步
################################################################################

async def awgSync(measure):
    t_list = measure.t_list
    wf = WF(t_list)
    awg_list = measure.awg
    for i in awg_list:
        wait = 'ITR' if awg_list[i] == awg_list['awg_trig'] else 'ATR'
        await awg_list[i].create_waveform(name=''.join((i,'_sync1')),length=len(measure.t_list),format=None)
        await awg_list[i].create_sequence(name=''.join((i,'_syncSeq')),steps=3,tracks=1)
        height,t_end,width = [1],[50000],[2000]
        sample = wf.square_wave(t_end,width,height)
        await awg_list[i].update_waveform(sample,name=''.join((i,'_sync1')))
        for j in range(3):
            j += 1
            goto = 'FIRST' if j == 3 else 'NEXT'
            await awg_list[i].set_sequence_step(name=''.join((i,'_syncSeq')),sub_name=[''.join((i,'_sync1'))],\
                                                   step=j,wait=wait,goto=goto,repeat=1,jump=None)
        await awg_list[i].use_sequence(name=''.join((i,'_syncSeq')),channels=[1])
        await awg_list[i].query('*OPC?')
        await awg_list[i].output_on(ch=1)
        await awg_list[i].setValue('Force Jump',1,ch=1)
        await awg_list[i].run()  

################################################################################
# 等效磁通计算
################################################################################

def voltTophi(qubit,bias,distance=0,freq=0):
    tm = np.arange(-qubit.T_bias[0]/2,0,0.001)
    t = tm + qubit.T_bias[1] if qubit.T_bias[1] > bias else -tm + qubit.T_bias[1]
    x, func = sy.Symbol('x'), qubit.specfunc
    y = sy.lambdify(x,func,'numpy')
    spec = y(t)
    bias_deviate = t[np.abs(spec-freq).argmin()] - bias
    f_bias = y(bias)
    f_distance = y(bias+distance)
    f_deviate = f_distance - f_bias
    return f_bias, f_deviate, bias_deviate

################################################################################
# 腔频率设置
################################################################################

async def resn(f_cavity):
    f_lo = f_cavity.max() + 50e6
    delta =  f_lo - f_cavity 
    n = len(f_cavity)
    return f_lo, delta, n

################################################################################
# 直流源设置
################################################################################

async def dcManage(measure,dcstate={},readstate=[],calimatrix=None,qnum=10):
    matrix = np.mat(np.eye(qnum)) if np.all(calimatrix) == None else np.mat(calimatrix)
    bias = [0] * qnum
    fread = []
    qubitToread = []
    for i,j in enumerate(measure.freqall):
        bias[i] = dcstate[j] if j in dcstate else 0
        if readstate != None:
            if readstate == []:
                if j in dcstate:
                    qubitToread.append(j)
                    fread.append(measure.freqall[j])
            else:
                if j in readstate: 
                    qubitToread.append(j)
                    fread.append(measure.freqall[j])
    if readstate != None:
        measure.qubitToread = qubitToread
        f_lo, delta, n = await resn(np.array(fread))
        measure.n, measure.delta, measure.f_lo = n, delta, f_lo
        await cw.modulation_read(measure,delta,tdelay=measure.readlen)
        await cw.couldRun(measure.awg['awgread'])
    current = matrix.I * np.mat(bias).T
    for i,j in enumerate(measure.freqall):
        await measure.dc[j].DC(round(current[i,0],3))

    # return f_lo, delta, n

################################################################################
# 激励频率计算
################################################################################

async def exMixing(f):
    if f == {}:
        return 
    qname = [i for i in f]
    f_ex = np.array([f[i] for i in f])
    ex_lo = f_ex.max() + 110e6
    delta =  ex_lo - f_ex
    delta_ex = {qname[i]:delta[i] for i in range(len(qname))}
    # n = len(f_ex)
    return ex_lo, delta_ex

################################################################################
# 激励源设置
################################################################################

async def exManage(measure,dcstate={},exstate={},calimatrix=None,qnum=10):
    qubits = measure.qubits
    matrix = np.mat(np.eye(qnum)) if np.all(calimatrix) == None else np.mat(calimatrix)
    bias = [0] * qnum
    f_ex1, f_ex2, f_ex3 = {}, {}, {}
    delta_ex1, delta_ex2, delta_ex3 = {}, {}, {}
    for i,j in enumerate(qubits):
        bias[i] = dcstate[j] if j in dcstate else 0
        if j in exstate:
            Att_Setup(measure,qubits[j].inst['com']).Att(exstate[j])
            if qubits[j].inst['ex_lo'] == 'psg_ex1':
                f_ex1[j] = qubits[j].f_ex[0]
            if qubits[j].inst['ex_lo'] == 'psg_ex2':
                f_ex2[j] = qubits[j].f_ex[0]
            if qubits[j].inst['ex_lo'] == 'psg_ex3':
                f_ex3[j] = qubits[j].f_ex[0]
    if f_ex1 != {}:
        ex_lo1, delta_ex1 = await exMixing(f_ex1)
        await measure.psg['psg_ex1'].setValue('Frequency',ex_lo1)
        await measure.psg['psg_ex1'].setValue('Output','ON')
    if f_ex2 != {}:
        ex_lo2, delta_ex2 = await exMixing(f_ex2)
        await measure.psg['psg_ex2'].setValue('Frequency',ex_lo2)
        await measure.psg['psg_ex2'].setValue('Output','ON')
    if f_ex3 != {}:
        ex_lo3, delta_ex3 = await exMixing(f_ex3)
        await measure.psg['psg_ex3'].setValue('Frequency',ex_lo3)
        await measure.psg['psg_ex3'].setValue('Output','ON')
    delta_ex = {**delta_ex1,**delta_ex2,**delta_ex3}
    current = matrix.I * np.mat(bias).T
    current = {f'q{i+1}':current[i,0] for i in range(qnum)}
    return delta_ex, current

################################################################################
# 拉比测试
################################################################################

async def rabitest(measure,t_rabi,comwave,amp=1,nwave=1,exstate={}):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    delta_ex, current = await exManage(measure,exstate=exstate)
    bit = measure.qubits
#     for i in delta_ex:
#         qubit = bit[i]
#         name = ''.join((qubit.inst['ex_awg'],qubit.q_name,'rabi'))
#         awg = measure.awg[qubit.inst['ex_awg']]
#         await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
#         await cw.genSeq(measure,awg,name)
#         if comwave:
#             await cw.Rabi_sequence(measure,name,awg,t_rabi,amp,nwave,delta_lo=delta_ex[i])
#         await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
# 并行计算取代以上注释部分
    task1, task2 = [], []
    for i in delta_ex:
        qubit = bit[i]
        name = ''.join((qubit.inst['ex_awg'],qubit.q_name,'rabi'))
        awg = measure.awg[qubit.inst['ex_awg']]
        namelist.append(name)
        awglist.append(awg)
        await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
        await cw.genSeq(measure,awg,name)
        if comwave:
            task_ready = cw.Rabi_sequence(measure,name,awg,t_rabi,amp,nwave,delta_lo=delta_ex[i])
            task.append(task_ready)
        task_manageawg = cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
        task2.append(task_manegeawg)
    concurrence([task1,task2])
# 中间部分
    measure.wave['Read'] = [['Readout_I']*len(t_rabi),['Readout_Q']*len(t_rabi)]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    # zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    # zname = [qubit.q_name+'_z']
    # measure.wave[zseqname] = [zname*len_data]
    # await cw.genwaveform(z_awg,zname,qubit.inst['z_ch'])
    # await cw.zWave(z_awg,zname,qubit.inst['z_ch'],0)
    # await cw.readSeq(measure,z_awg,zseqname,qubit.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# 优化读出点测试
################################################################################

async def readOptest(measure,modulation=False,freq=None,exstate={}):
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    delta_ex, current = await exManage(measure,exstate=exstate)
    bit = measure.qubits
    for i in delta_ex:
        qubit = bit[i]
        pilen = qubit.pi_len
        awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

        await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])
        pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9,Delta_lo=delta_ex[i],DRAGScaling=None)
        await cw.writeWave(awg,qname,pulse)
        await cw.couldRun(awg)
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        job = Job(S21, (qubit,measure,modulation,freq),auto_save=False,max=126)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [j]*n, f_s21, s_s21

################################################################################
# 临界判断测试
################################################################################

async def threshHoldtest(meausre,modulation=True,exstate={}):
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    delta_ex, current = await exManage(measure,exstate=exstate)
    bit = measure.qubits
    if modulation:
        for i in delta_ex:
            qubit = bit[i]
            pilen = qubit.pi_len
            awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    
            await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])
            pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9,Delta_lo=delta_ex[i])
            await cw.writeWave(awg,qname,pulse)
            await cw.couldRun(awg)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for j, i in enumerate(['OFF','ON']):
        for k in delta_ex:
            qubit = bit[k]
            await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        time.sleep(2)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield j, s

################################################################################
# 开关JPA
################################################################################

async def jpa_switch(measure,state='OFF'):
    if state == 'ON':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','ON')
        await measure.psg[measure.jpa.inst['pump']].setValue('Frequency',(measure.jpa.f_ex))
        await measure.psg[measure.jpa.inst['pump']].setValue('Power',measure.jpa.power_ex)
        await measure.dc[measure.jpa.q_name].DC(measure.jpa.bias)
    if state == 'OFF':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','OFF')
        await  measure.dc[measure.jpa.q_name].DC(0)

################################################################################
# 关闭仪器
################################################################################

async def close(measure):
    inst = {**measure.dc , **measure.psg}
    for i in inst:
        await inst[i].setValue('Output','OFF')

################################################################################
# 查询仪器状态
################################################################################

async def QueryInst(measure):
    inst = {**measure.dc , **measure.psg}
    state = {}
    for i in inst:
        try:
            if 'psg' in i:
                freq = await inst[i].getValue('Frequency')
                power = await inst[i].getValue('Power')
                Output = await inst[i].getValue('Output')
                Moutput = await inst[i].getValue('Moutput')
                Mform = await inst[i].getValue('Mform')
                err = (await inst[i].query('syst:err?')).strip('\n\r').split(',')
                sm = {'freq':'%fGHz'%(freq/1e9),'power':'%fdBm'%power,'output':Output.strip('\n\r'),\
                    'moutput':Moutput.strip('\n\r'),'mform':Mform.strip('\n\r'),'error':err[0]}
                state[i] = sm
            else:
                current = await inst[i].getValue('Offset')
                load = await inst[i].getValue('Load')
                load = eval((load).strip('\n\r'))  
                load = 'high Z' if load != 50 else 50
                err = (await inst[i].query('syst:err?')).strip('\n\r').split(',')
                sm = {'offset':current,'load':load,'error':err[0]}
                state[i] = sm
        finally:
            pass
    measure.inststate = state
    return state

################################################################################
# 初始化仪器
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
            measure.wave = {}

################################################################################
# 恢复仪器最近状态
################################################################################

async def RecoverInst(measure,state=None):
    if state is None:
        state = measure.inststate
    for i in state:
        if 'psg' in i:
            await measure.psg[i].setValue('Frequency',eval(state[i]['freq'].strip('GHz'))*1e9)
            await measure.psg[i].setValue('Power',eval(state[i]['power'].strip('dBm')))
            if state[i]['output'] == '1':
                await measure.psg[i].setValue('Output','ON')
            else:
                await measure.psg[i].setValue('Output','OFF')
        else:
            await measure.dc[i].DC(state[i]['offset'])

################################################################################
# S21
################################################################################

async def S21(qubit,measure,modulation=False,f_lo=None,f=None):
    #await jpa_switch(measure,state='OFF')
    if f_lo == None:
        f_lo, delta, n = await resn(np.array(qubit.f_lo))
        freq = np.linspace(-2.5,2.5,201)*1e6 + f_lo
    else:
        freq = np.linspace(-2.5,2.5,201)*1e6 + f_lo
        delta, n = measure.delta, measure.n
    if modulation:
        await cw.modulation_read(measure,delta,tdelay=measure.readlen)
    await cw.couldRun(measure.awg['awgread'])
    await measure.psg['psg_lo'].setValue('Output','ON')
    if f is not None:
        freq = f
    for i in freq:
        await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield i-delta, s

async def test(measure,n):
    for i in range(n):
        time.sleep(0.1)
        # await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s

################################################################################
# 读出信号相位噪声
################################################################################

async def rPhase(measure,phase):
    await measure.psg['psg_lo'].setValue('Output','ON')
    for i in phase:
        await cw.modulation_read(measure,measure.delta,tdelay=1200,phase=i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s

################################################################################
# 重新混频
################################################################################

async def again(qubit,measure,modulation=False,flo=None,freq=None):
    #f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    #freq = np.linspace(-2.5,2.5,126)*1e6+f_lo
    for i in measure.psg:
        if i != 'psg_lo' and i != 'psg_pump':
            await measure.psg[i].setValue('Output','OFF')
    length = len(freq) if freq is not None else 201
    job = Job(S21, (qubit,measure,modulation,flo,freq),auto_save=True,max=length,tags=[qubit.q_name])
    f_s21, s_s21 = await job.done()
    index = np.abs(s_s21).argmin(axis=0)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n = await resn(np.array(f_res))
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        #await cw.ats_setup(measure.ats,delta)
        await cw.modulation_read(measure,delta,tdelay=measure.readlen)
        base = 0
        for i in range(15):
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            base += Am + 1j*Bm
        base /= 15
    measure.base, measure.n, measure.delta, measure.f_lo = base, n, delta, np.array([f_lo])
    return f_lo, delta, n, f_res, base,f_s21, s_s21

################################################################################
# S21vsFlux
################################################################################

async def S21vsFlux(qubit,measure,current,calimatrix,modulation=False):
    await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    for i in current:
        await dcManage(measure,dcstate={qubit.q_name:i},readstate=None,calimatrix=calimatrix)
        # await measure.dc['q2'].DC(5)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# S21vsFlux_awgoffset
################################################################################

async def S21vsFlux_awgoffset(qubit,measure,current,calimatrix,modulation=False):
    awg = measure.awg[qubit.inst['z_awg']]
    qname = [''.join((qubit.q_name,'_z'))]
    await cw.genwaveform(awg,qname,qubit.inst['z_ch'])
    await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    for i in current:
        await cw.zWave(awg,qname,volt=0,offset=i)
        await cw.couldRun(awg)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# S21vsPower
################################################################################

async def S21vsPower(qubit,measure,att,com='com8',modulation=False):
    l = measure.readlen
    await dcManage(measure,dcstate={},readstate=[qubit.q_name],calimatrix=None)
    await cw.modulation_read(measure,measure.delta,tdelay=2000)
    measure.readlen = l
    for i in att:
        await measure.attinst['com8'].set_att(i)
        job = Job(S21, (qubit,measure,False,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# SingleSpec
################################################################################

async def singlespec(qubit,measure,freq,modulation=False,f_read=None,readponit=True):
    if readponit:
        f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,f_read)
    else:
        n, base = measure.n, measure.base
    await measure.psg['psg_trans'].setValue('Output','ON')
    await measure.psg['psg_trans'].setValue('Moutput','ON')
    for i in freq:
        await measure.psg['psg_trans'].setValue('Frequency',i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base
    await measure.psg['psg_trans'].setValue('Output','OFF')
################################################################################
# SingleSpec扫电压
################################################################################

async def specbias(qubit,measure,ftarget,bias,modulation=False):
    await measure.dc[qubit.q_name].DC(round(np.mean(bias),3))
    await measure.psg['psg_trans'].setValue('Frequency',ftarget)
    f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,measure.f_lo)
    await measure.psg['psg_trans'].setValue('Output','ON')
    for i in bias:
        await measure.dc[qubit.q_name].DC(i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*n, s-base
    await measure.psg['psg_trans'].setValue('Output','OFF')

################################################################################
# Spec2d
################################################################################

async def spec2d(qubit,measure,freq,calimatrix,modulation=False):
    current = np.linspace(-qubit.T_bias[0]*0.35,qubit.T_bias[0]*0.35,36) + qubit.T_bias[1] 
    await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    for i in current:
        await dcManage(measure,dcstate={qubit.q_name:i},readstate=None,calimatrix=calimatrix)
        # await measure.dc[qubit.q_name].DC(i)
        job = Job(singlespec, (qubit,measure,freq,modulation,measure.f_lo,True),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

################################################################################
# Rabi
################################################################################

async def rabi(qubit,measure,t_rabi,comwave=False,amp=1,nwave=1):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])

    name = ''.join((qubit.inst['ex_awg'],'coherence_rabi'))
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    if comwave:
        await cw.Rabi_sequence(measure,name,awg,t_rabi,amp,nwave)

    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
    measure.wave['Read'] = [['Readout_I']*len(t_rabi),['Readout_Q']*len(t_rabi)]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    zname = [qubit.q_name+'_z']
    measure.wave[zseqname] = [zname*len_data]
    await cw.genwaveform(z_awg,zname,qubit.inst['z_ch'])
    await cw.zWave(z_awg,zname,volt=0,offset=0)
    await cw.readSeq(measure,z_awg,zseqname,qubit.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# Rabi_waveform
################################################################################

async def Rabi_waveform(qubit,measure,t_rabi,nwave=1):

    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    for i in t_rabi:
        pulse = await cw.rabiWave(envelopename=measure.envelopename,nwave=nwave,\
            during=i/1e9,phase=0,phaseDiff=0,DRAGScaling=None)
        await cw.writeWave(awg,qname,pulse)
        await cw.couldRun(awg)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# Rabi_population
################################################################################

async def rabiPop(qubit,measure,t_rabi,which):
    # await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in t_rabi:
        await cw.rabiWave(awg,during=i/1e9,name=qname)
        await cw.couldRun(awg)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        ss = s[:,0]
        d = list(zip(np.real(ss),np.imag(ss)))
        y = measure.predict(d)
        pop = list(y).count(which)/len(y)
        yield [i], [pop]

################################################################################
# 优化pi脉冲
################################################################################
    
async def pipulseOpt(qubit,measure,nwave,wavlen):
    pilen = qubit.pi_len
    start = 0.5*pilen if 0.5*pilen > 10 else 1
    end = 2.5*pilen if 0.5*pilen > 10 else pilen+30
    t = np.linspace(start,end,wavlen)
    for i in range(nwave):
        job = Job(rabi, (qubit,measure,t,True,1,2*i+1), max=500,avg=True,auto_save=False)
        t_r, s_r = await job.done()
        yield [2*i+1]*measure.n, t_r, s_r

async def pipulseOpt_waveform(qubit,measure,nwave,t):
    # pilen = qubit.pi_len
    # t = np.arange(int(0.5*pilen),int(1.5*pilen),0.5)
    for i in range(nwave):
        job = Job(Rabi_waveform, (qubit,measure,t,2*i+1), max=len(t),auto_save=False)
        t_r, s_r = await job.done()
        yield [2*i+1]*measure.n, t_r, s_r

################################################################################
# pi脉冲振幅
################################################################################
    
async def pipulseAmp(qubit,measure,t_rabi):
    amplitude = np.linspace(0.2,1,9)
    for i in amplitude:
        job = Job(rabi, (qubit,measure,t_rabi,True,i,1), max=500,avg=True,auto_save=False)
        t_r, s_r = await job.done()
        yield [i]*measure.n, t_r, s_r

################################################################################
# 优化pi脉冲detune
################################################################################
    
async def detuneOpt(qubit,measure,which,alpha,nwave=10):
    pilen, freq = qubit.pi_len, np.linspace(-20,20,201)*1e6 + qubit.delta_ex[0]
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])
    # await cw.pipulseDetunewave(measure,awg,pilen,[[0,-np.pi],[2*pilen,0]],alpha,qname)
    # await cw.couldRun(awg)

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in freq:
        # await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',i)
        pulselist = await cw.pipulseDetunewave(measure,awg,pilen,nwave,alpha,i,qname)
        await cw.couldRun(awg)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[:,0], ch_B[:,0]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        d = list(zip(np.real(ss),np.imag(ss)))
        y = measure.predict[qubit.q_name](d)
        pop = list(y).count(which)/len(y)
        yield [i], [pop]

################################################################################
# 优化dragcoef
################################################################################

async def dragcoefHD(qubit,measure,which,alpha,axis=['Xnhalf','Xhalf'],nwave=1):
    pilen, coef = qubit.pi_len, np.linspace(-2,2,41)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for j in axis:
        for i in coef:
            await cw.HDWave(measure,awg,pilen,i/alpha,j,nwave,qname)
            await cw.couldRun(awg)
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            # pop.append(ss)
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[qubit.q_name](d)
            pop = list(y).count(which)/len(y)
            yield [i], [pop]

################################################################################
# AllXY drag detune
################################################################################
    
async def AllXYdragdetune(qubit,measure,which,alpha):
    pilen, coef = qubit.pi_len, np.linspace(-3,3,61)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for j in [['X','Yhalf'],['Y','Xhalf']]:
        for i in coef:
            await cw.dragDetunewave(measure,awg,pilen,i/alpha,j,qname)
            await cw.couldRun(awg)
            ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=True)
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[qubit.q_name](d)
            pop = list(y).count(which)/len(y)
            yield [i], [pop]
            # ch_A, ch_B, I, Q = await measure.ats.getIQ()
            # Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # # theta0 = np.angle(Am) - np.angle(Bm)
            # # Bm *= np.exp(1j*theta0)
            # s = Am + 1j*Bm
            # yield [i], s-measure.base

################################################################################
# 优化IQ-Mixer相位
################################################################################
    
async def IQphaseOpt(qubit,measure,which,nwave,alpha):
    pilen, angle = qubit.pi_len, np.linspace(-np.pi/2,np.pi/2,201)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in angle:
        pulselist = await cw.pipulseDetunewave(measure,awg,pilen,nwave,alpha,qubit.delta_ex[0],qname,i)
        await cw.couldRun(awg)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[:,0], ch_B[:,0]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        d = list(zip(np.real(ss),np.imag(ss)))
        y = measure.predict[qubit.q_name](d)
        pop = list(y).count(which)/len(y)
        yield [i], [pop]

################################################################################
# Rabi_power
################################################################################

async def rabiPower(qubit,measure,t_rabi=20,comwave=False,amp=1,nwave=1):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])

    name = ''.join((qubit.inst['ex_awg'],'coherence_rabi'))
    len_data, t = len(amp), np.array([amp]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(amp),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    if comwave:
        await cw.Rabipower_sequence(measure,name,awg,t_rabi,amp,nwave)

    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
    measure.wave['Read'] = [['Readout_I']*len(amp),['Readout_Q']*len(amp)]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    # zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    # zname = [qubit.q_name+'_z']
    # measure.wave[zseqname] = [zname*len_data]
    # await cw.genwaveform(z_awg,zname,qubit.inst['z_ch'])
    # await cw.zWave(z_awg,zname,qubit.inst['z_ch'],0)
    # await cw.readSeq(measure,z_awg,zseqname,qubit.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# IQ-Mixer优化
################################################################################

async def optIQMixer(qubit,measure,att):
    name, awg = [qubit.q_name+'_I', qubit.q_name+'_Q'], measure.awg[qubit.inst['ex_awg']]
    await cw.genwaveform(awg,name,qubit.inst['ex_ch'])
    # await cw.rabiWave(awg,during=25e-9,name=name)
    await cw.couldRun(awg)
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',(qubit.f_ex)[0])
    await measure.psg['psg_lo'].setValue('Output','ON')
    n = len(measure.base)

    for i in att:
        # att_setup.Att(i,False)
        await measure.psg[qubit.inst['ex_lo']].setValue('Power',i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i]*n, s - measure.base
    # att_setup.close()

################################################################################
# Z_Pulse（Z与Readout时序矫正)
################################################################################

async def singleZpulse(qubit,measure,t_rabi,Delta_lo,comwave=True,offset=0):
    len_data = len(t_rabi)
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
    measure.wave['Read'] = [['Readout_I']*len_data,['Readout_Q']*len_data]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    name =  ''.join((qubit.inst['ex_awg'],'coherence')) #name--kind
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len_data,len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    if comwave:
        await cw.T1_sequence(measure,name,awg,t_rabi,qubit.pi_len,0,Delta_lo)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])

    zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    zname = [qubit.q_name+'_z']
    measure.wave[zseqname] = [zname*len_data]
    await cw.genwaveform(z_awg,zname,qubit.inst['z_ch'])
    pulse1 = await cw.zWave(z_awg,zname,volt=-0.1,during=1000e-9,shift=1000e-9)
    pulse2 = await cw.zWave(z_awg,zname,volt=offset,during=1600e-9,shift=500e-9)
    pulselist = np.array(pulse1) + np.array(pulse2)
    await cw.writeWave(z_awg,zname,pulselist)
    await cw.readSeq(measure,z_awg,zseqname,qubit.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    t = np.array([t_rabi]*measure.n).T
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

async def zPulse_offset(qubit,measure,t):

    offset = np.linspace(0,0.12,25)
    for j,i in enumerate(offset):
        comwave = True if j == 0 else False
        job = Job(singleZpulse, (qubit,measure,t,qubit.delta_ex[0],comwave,i),no_bar=True,tags=[qubit.q_name], max=500,avg=True)
        t_AC, pop = await job.done()
        yield [i]*measure.n, t_AC, pop

async def zPulse_freq(qubit,measure,t):
    pilen = qubit.pi_len
    freq = np.linspace(-100,100,201)*1e6 + qubit.f_ex[0]

    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,volt=0.1,during=1000e-9,shift=1000e-9)
    await cw.couldRun(measure.awg[qubit.inst['z_awg']])

    for j in t:
        await cw.modulation_ex(qubit,measure,2*pilen/1e9,shift=(j-500)/1e9)
        job = Job(singlespec, (qubit,measure,freq,False,None,False),max=len(freq))
        f, s = await job.done()
        yield [j]*measure.n, f, s

################################################################################
# ramsey_Z_Pulse（Z与Readout时序矫正)
################################################################################

async def ramseyZpulse(qubit,measure,tdelay,which,tcali=0,DRAGScaling=None,nozpulse=False):
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    ex_name = [qubit.q_name+'_I', qubit.q_name+'_Q']

    await cw.genwaveform(awg,ex_name,qubit.inst['ex_ch'])
    n, pilen = len(measure.delta), qubit.pi_len
    for i in tdelay:
        pulse = await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,volt=0.1,during=500e-9,\
            shift=(i+tcali+qubit.pi_len)/1e9)
        await cw.couldRun(measure.awg[qubit.inst['z_awg']])
        pop = []
        for axis in ['Ynhalf','Xhalf']:
            await cw.ramseyZwave(measure,awg,pilen,axis,ex_name,DRAGScaling,shift=qubit.timing['read>xy'])  
            await cw.couldRun(awg)
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            # pop.append(ss)
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[qubit.q_name](d)
            pop.append(list(y).count(which)/len(y))
        yield [i], pop

################################################################################
# ramsey_Z_Pulse_chen
################################################################################

async def ramseyZpulse_chen(qubit,measure,tdelay,which,tcali=0,DRAGScaling=None,nozpulse=False):
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    ex_name = [qubit.q_name+'_I', qubit.q_name+'_Q']

    await cw.genwaveform(awg,ex_name,qubit.inst['ex_ch'])
    n, pilen = len(measure.delta), qubit.pi_len
    volt = 0 if nozpulse else 0.1
    for i in tdelay:
        pulse = await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,volt=volt,during=500e-9,shift=(i+tcali+pilen)/1e9)
        await cw.couldRun(measure.awg[qubit.inst['z_awg']])
        pop = []
        for axis in ['Ynhalf','Xhalf']:
            await cw.ramseyZwave_chen(measure,awg,pilen,axis,(i+qubit.timing['read>xy'])/1e9,ex_name,DRAGScaling)  
            await cw.couldRun(awg)
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            # pop.append(ss)
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[qubit.q_name](d)
            pop.append(list(y).count(which)/len(y))
        yield [i], pop
    

################################################################################
# Z_Pulse_XY（Z与Readout时序矫正) population
################################################################################

async def zPulse_XY(qubit,measure,tdelay,which,tcali=0,DRAGScaling=None):
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    ex_name = [qubit.q_name+'_I', qubit.q_name+'_Q']

    n, pilen = len(measure.delta), qubit.pi_len
    await cw.genwaveform(awg,ex_name,qubit.inst['ex_ch'])
    await cw.ramseyZwave(measure,awg,pilen,'Yhalf',ex_name,DRAGScaling)  
    await cw.couldRun(awg)
    for i in tdelay:
        pulse = await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,volt=0.2,during=500e-9,shift=(i+tcali)/1e9)
        await cw.couldRun(measure.awg[qubit.inst['z_awg']])
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[:,0],ch_B[:,0]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        d = list(zip(np.real(ss),np.imag(ss)))
        y = measure.predict[qubit.q_name](d)
        pop = list(y).count(which)/len(y)
        yield [i], [pop]

################################################################################
# AC-Stark（XY与Readout时序矫正)
################################################################################

async def singleacStark(qubit,measure,t_rabi,Delta_lo=80e6):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
    measure.wave['Read'] = [['Readout_I']*len(t_rabi),['Readout_Q']*len(t_rabi)]
    await cw.ac_stark_wave(measure)
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    name =  ''.join((qubit.inst['ex_awg'],'coherence')) #name--kind
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    await cw.T1_sequence(measure,name,awg,t_rabi,qubit.pi_len,0,Delta_lo)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])

    zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    zname = [qubit.q_name+'_z']
    measure.wave[zseqname] = [zname*len_data]
    await cw.genwaveform(z_awg,zname,qubit.inst['z_ch'])
    await cw.zWave(z_awg,zname,qubit.inst['z_ch'],volt=0,offset=0)
    await cw.readSeq(measure,z_awg,zseqname,qubit.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base
    
async def acStark(qubit,measure,t):
    pilen = qubit.pi_len
    freq = np.linspace(-100,100,201)*1e6 + qubit.f_ex[0]

    await cw.ac_stark_wave(measure)
    await cw.couldRun(measure.awg['awgread'])

    zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,volt=0,during=1000e-9,shift=1000e-9)
    await cw.couldRun(measure.awg[qubit.inst['z_awg']])

    for j in t:
        await cw.modulation_ex(qubit,measure,2*pilen/1e9,shift=(j-500)/1e9)
        job = Job(singlespec, (qubit,measure,freq,False,None,False),max=len(freq))
        f, s = await job.done()
        yield [j]*measure.n, f, s

################################################################################
# 优化读出点
################################################################################

async def readOp(qubit,measure,modulation=False,freq=None):
    pilen = qubit.pi_len
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])

    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])
    pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9,amp=0.265,shift=qubit.timing['read>xy']/1e9,DRAGScaling=None)
    await cw.writeWave(awg,qname,pulse)
    await cw.couldRun(awg)
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo),auto_save=False,max=201)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [j]*n, f_s21, s_s21

################################################################################
# 优化读出长度
################################################################################

async def readWavelen(qubit,measure):
    pilen, t = qubit.pi_len, np.linspace(900,2000,21,dtype=np.int64)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])
    pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9)
    await cw.writeWave(awg,qname,pulse)
    await cw.couldRun(awg)

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])

    for k in t:
        await cw.modulation_read(measure,measure.delta,tdelay=int(k),repeats=5000)
        state = []
        for j, i in enumerate(['OFF','ON']):
            await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            state.append((j,np.mean(s),np.std(s)))
        yield k, state

################################################################################
# 临界判断
################################################################################

async def threshHold(qubit,measure,modulation=True):
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])
    
    if modulation:
        pilen = qubit.pi_len
        awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    
        await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])
        pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9,shift=qubit.timing['read>xy']/1e9,DRAGScaling=None)
        await cw.writeWave(awg,qname,pulse)
        await cw.couldRun(awg)
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        time.sleep(2)
        ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=True)
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        # s = Am + Bm
        s = Am + 1j*Bm
        sq = I + 1j*Q
        yield j, s, sq

################################################################################
# 临界判断3态
################################################################################

async def threshHold3(qubit,measure,modulation=True):
    f02 = qubit.f_ex*2 - qubit.alpha
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])
    
    if modulation:
        pilen = qubit.pi_len
        await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
        awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    
        await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])
        pulse = await cw.rabiWave(envelopename=measure.envelopename,during=pilen/1e9,DRAGScaling=None)
        await cw.writeWave(awg,qname,pulse)
        await cw.couldRun(awg)
    for j, i in enumerate(['OFF','ON']):
        await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
        time.sleep(2)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield j, s

################################################################################
# 优化读出振幅
################################################################################

async def readpowerOpt(qubit,measure,com,att):
    att_setup = Att_Setup(measure,com)
    n = len(measure.delta)
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in att:
        att_setup.Att(i,False)
        job = Job(threshHold,(qubit,measure,False),auto_save=False,no_bar=True)
        st, s_st = await job.done()
        x, y = s_st[0], s_st[1]
        offmean, onmean = np.mean(x), np.mean(y)
        offstd, onstd = np.std(x), np.std(y)
        # doff, don = np.abs(x-offmean), np.abs(y-onmean)
        d = np.abs(offmean-onmean)
        # popoff, popon = d/offstd, d/onstd
        snr = 2*d**2/(offstd+onstd)**2
        # popoff, popon = list(doff<offstd).count(True)/len(x), list(don<onstd).count(True)/len(y)
        # yield [i]*n, [popoff]*n, [popon]*n
        yield [i]*n, [snr]*n
    att_setup.close()

################################################################################
# T1
################################################################################

async def T1(qubit,measure,t_rabi,comwave=False,clearseq=True):
    if clearseq:
        await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
        measure.wave['Read'] = [['Readout_I']*len(t_rabi),['Readout_Q']*len(t_rabi)]
        await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    name = ''.join((qubit.inst['ex_awg'],'coherence_T1'))
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    if comwave:
        await cw.T1_sequence(measure,name,awg,t_rabi,qubit.pi_len)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
    await cw.couldRun(awg)
    
    zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    zname = [qubit.q_name+'_z']
    measure.wave[zseqname] = [zname*len_data]
    await cw.genwaveform(z_awg,zname,qubit.inst['z_ch'])
    await cw.zWave(z_awg,zname,volt=0,offset=0)
    await cw.readSeq(measure,z_awg,zseqname,qubit.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# Ramsey
################################################################################

async def Ramsey(qubit,measure,t_rabi,comwave=False):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
    measure.wave['Read'] = [['Readout_I']*len(t_rabi),['Readout_Q']*len(t_rabi)]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    name = ''.join((qubit.inst['ex_awg'],'coherence'))
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    if comwave:
        await cw.Coherence_sequence(measure,name,awg,t_rabi,qubit.pi_len,n_wave=0,seqtype='PDD')
        # await cw.Ramsey_sequence(measure,name,awg,t_rabi,qubit.pi_len)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])
    await cw.couldRun(awg)

    zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    zname = [qubit.q_name+'_z']
    measure.wave[zseqname] = [zname*len_data]
    await cw.genwaveform(z_awg,zname,qubit.inst['z_ch'])
    await cw.zWave(z_awg,zname,volt=0,offset=0)
    await cw.readSeq(measure,z_awg,zseqname,qubit.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

async def Ramsey_waveform(qubit,measure,t_rabi):

    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    halfpi = qubit.pi_len
    for i in t_rabi:
        pulse = await cw.coherenceWave(measure.envelopename,i/1e9,halfpi/1e9,n_wave=0,seqtype='PDD')
        await cw.writeWave(awg,qname,pulse)
        await cw.couldRun(awg)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# 真空拉比
################################################################################

async def vRabi(qubit,measure,t_rabi,v_rabi,clearseq=True):
    if clearseq:
        await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
        measure.wave['Read'] = [['Readout_I']*len(t_rabi),['Readout_Q']*len(t_rabi)]
        await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])
    zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    # zname = [qubit.q_name+'_z']
    await cw.create_wavelist(measure,zseqname,(z_awg,['step'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,z_awg,zseqname)
    await cw.awgchmanage(z_awg,zseqname,qubit.inst['z_ch'])
    await cw.couldRun(z_awg)
    
    timing = (qubit.timing['z>xy'] - qubit.timing['read>xy'])*1e-9
    for i, j in enumerate(v_rabi):
        await cw.Z_sequence(measure,zseqname,z_awg,t_rabi,volt=j,shift=timing)
        comwave = True if i == 0 else False
        job = Job(T1, (qubit,measure,t_rabi,comwave,False), tags=[qubit.q_name], max=500,avg=True, no_bar=True)
        t_t, s_t = await job.done()
        clear_output()
        yield [j]*measure.n, t_t, s_t

################################################################################
# SpinEcho
################################################################################

async def SpinEcho(qubit,measure,t_rabi,n_wave=0,seqtype='CPMG',comwave=False):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
    measure.wave['Read'] = [['Readout_I']*len(t_rabi),['Readout_Q']*len(t_rabi)]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    name = ''.join((qubit.inst['ex_awg'],'coherence'))
    awg = measure.awg[qubit.inst['ex_awg']]
    await cw.create_wavelist(measure,name,(awg,['I','Q'],len(t_rabi),len(measure.t_list)))
    await cw.genSeq(measure,awg,name)
    if comwave:
        await cw.Coherence_sequence(measure,name,awg,t_rabi,qubit.pi_len,n_wave,seqtype)
    await cw.awgchmanage(awg,name,qubit.inst['ex_ch'])

    zseqname, z_awg = qubit.inst['z_awg']+'_step', measure.awg[qubit.inst['z_awg']]
    zname = [qubit.q_name+'_z']
    measure.wave[zseqname] = [zname*len_data]
    await cw.genwaveform(z_awg,zname,qubit.inst['z_ch'])
    await cw.zWave(z_awg,zname,volt=0,offset=0)
    await cw.readSeq(measure,z_awg,zseqname,qubit.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# Ramsey crosstalk_cali
################################################################################

async def Z_cross(qubit_ex,qubit_z,measure,v_rabi,len_data,comwave=False):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
    measure.wave['Read'] = [['Readout_I']*len(v_rabi),['Readout_Q']*len(v_rabi)]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    t, name_ex, name_z = v_rabi[1:len_data+1], ''.join((qubit_ex.inst['ex_awg'],'coherence')), ''.join((qubit_z.inst['z_awg'],'_z'))
    t = np.array([t]*measure.n).T
    ex_awg, z_awg = measure.awg[qubit_ex.inst['ex_awg']], measure.awg[qubit_z.inst['z_awg']]

    await cw.create_wavelist(measure,name_z,(z_awg,['Z'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,z_awg,name_z)
    await cw.create_wavelist(measure,name_ex,(ex_awg,['I','Q'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,ex_awg,name_ex)
    if comwave:
        await cw.Z_cross_sequence(measure,name_z,name_ex,z_awg,ex_awg,v_rabi,qubit_ex.pi_len)
    await cw.awgchmanage(ex_awg,name_ex,qubit_ex.inst['ex_ch'])
    await cw.awgchmanage(z_awg,name_z,qubit_z.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,qubit_ex.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# spec crosstalk cali
################################################################################

async def single_cs(measure,t,len_data):
    for i in range(500):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

async def crosstalkSpec(qubit_ex,qubit_z,measure,v_rabi,len_data,comwave):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])
    measure.wave['Read'] = [['Readout_I']*len(v_rabi),['Readout_Q']*len(v_rabi)]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    t, name_ex, name_z = v_rabi[1:len_data+1], ''.join((qubit_ex.inst['ex_awg'],'coherence')), ''.join((qubit_ex.inst['z_awg'],'_z'))
    t = np.array([t]*measure.n).T
    ex_awg, zz_awg, exz_awg = measure.awg[qubit_ex.inst['ex_awg']], measure.awg[qubit_z.inst['z_awg']]\
        , measure.awg[qubit_ex.inst['z_awg']]
    await cw.create_wavelist(measure,name_z,(qubit_ex.inst['z_awg'],['Z'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,exz_awg,name_z)
    await cw.create_wavelist(measure,name_ex,(qubit_ex.inst['ex_awg'],['I','Q'],len(v_rabi),len(measure.t_list)))
    await cw.genSeq(measure,ex_awg,name_ex)
    await cw.genwaveform(zz_awg,[''.join((qubit_z.q_name,'_z'))],qubit_z.inst['z_ch'])

    if comwave:
        await cw.Speccrosstalk_sequence(measure,name_z,name_ex,exz_awg,ex_awg,v_rabi,400)

    await cw.awgchmanage(ex_awg,name_ex,qubit_ex.inst['ex_ch'])
    await cw.awgchmanage(exz_awg,name_z,qubit_ex.inst['z_ch'])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,qubit_ex.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in np.linspace(-1,1,11):
        await zz_awg.stop()
        await cw.zWave(measure,zz_awg,[''.join((qubit_z.q_name,'_z'))],qubit_z.inst['z_ch'],i,during=1100e-9,shift=200*1e-9)
        await zz_awg.run()
        await cw.couldRun(zz_awg)
        job = Job(single_cs,(measure,t,len_data),avg=True,max=500,auto_save=False,no_bar=True)
        v_bias, s =await job.done()
        n = np.shape(s)[1]
        yield [i]*n, v_bias, s

################################################################################
# 直流源crosstalk矫正
################################################################################

async def dcCrosstalk(q_target,q_bias,measure,deviate=0.1):
    measure.delta,measure.base,measure.readlen = q_target.state[1:]
    await RecoverInst(measure,state=q_target.state[0])
    await measure.dc[q_bias.q_name].DC(deviate)
    freq = np.linspace(-200,200,401)*1e6 + q_target.f_ex[0]
    job = Job(singlespec, (q_target,measure,freq,True,None,True), tags=[q_target.q_name,q_bias.q_name], max=len(freq))
    f_ss, s_ss = await job.done()
    fitdata = op.Lorentz_Fit().fitLorentz(f_ss[:,0]/1e9,np.abs(s_ss[:,0]))
    f_com = voltTophi(q_target,q_target.bias,freq=fitdata[1])
    print(f_com)
    ratio = deviate/f_com[2]
    bias_real = np.arange(-0.02,0.02,0.002) + 0.005*ratio

    await RecoverInst(measure,state=q_target.state[0])
    await measure.dc[q_target.q_name].DC(round(q_target.bias+0.005,3))
    await measure.psg['psg_trans'].setValue('Frequency',q_target.f_ex[0])
    # f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(q_target,measure,False)
    await measure.psg['psg_trans'].setValue('Output','ON')
    for i in bias_real:
        await measure.dc[q_bias.q_name].DC(i)
        # f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(q_target,measure,False)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        #theta = np.angle(s) - np.angle(base)
        #base *= np.exp(1j*theta)
        yield [i]*measure.n, s-measure.base
    await measure.psg['psg_trans'].setValue('Output','OFF')

################################################################################
# Randomized Benchmarking waveform
################################################################################

async def RB_waveform(qubit,measure,mlist,len_data,which,delta_ex,DRAGScaling=None,gate=None):
    
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])

    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for j in tqdm(mlist,desc='RB'):
        pop = []
        for i in range(len_data):
            pulse = await cw.rbWave(measure,j,gate,qubit.pi_len*1e-9,Delta_lo=delta_ex,DRAGScaling=DRAGScaling)
            await cw.writeWave(awg,qname,pulse)
            await cw.couldRun(awg)
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            # pop.append(ss)
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[qubit.q_name](d)
            pop.append(list(y).count(which)/len(y))
        yield [j], pop
    
################################################################################
# Tomo
################################################################################

async def tomo(qubit,measure,t_rabi,which,DRAGScaling):
    pilen = qubit.pi_len
    await cw.modulation_read(measure,measure.delta,tdelay=measure.readlen)
    awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']
    await cw.genwaveform(awg,qname,qubit.inst['ex_ch'])

    # zname, zch = [''.join((qubit.q_name,'_z'))], qubit.inst['z_ch']
    # await cw.genwaveform(measure.awg[qubit.inst['z_awg']],zname,zch)
    # await cw.zWave(measure.awg[qubit.inst['z_awg']],zname,zch,during=2000e-9,volt=0,offset=0)
    # await cw.couldRun(measure.awg[qubit.inst['z_awg']])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=5000)
    for i in t_rabi:
        pop = []
        for axis in ['Ynhalf','Xhalf','Z']:
            await cw.tomoTest(measure,awg,i,pilen,axis,qname,DRAGScaling)  
            await cw.couldRun(awg)
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            # pop.append(ss)
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[qubit.q_name](d)
            pop.append(list(y).count(which)/len(y))
        yield [i], pop
    
################################################################################
# RTO
################################################################################

async def RTO(qubit,measure,t):
    await cw.clearSeq(measure,['awg131','awg132','awg133','awg134'])

    name_ex = ''.join((qubit.inst['ex_awg'],'rto'))
    awg, len_data = qubit.inst['ex_awg'], 2*t
    await cw.create_wavelist(measure,name_ex,(awg,['I','Q'],2,len(measure.t_list)))
    measure.wave[rtolist] = [measure.wave[name_ex][0]*t,measure.wave[name_ex][1]*t]
    await cw.rtoWave(measure,awg,qubit.pi_len,[measure.wave[name_ex][0],measure.wave[name_ex][1]])
    await cw.genSeq(measure,awg,'rtolist')
    await cw.awgchmanage(awg,'rtolist',qubit.inst['ex_ch'])

    measure.wave['Read'] = [['Readout_I']*len_data,['Readout_Q']*len_data]
    await cw.readSeq(measure,measure.awg['awgread'],'Read',[1,5])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cw.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len_data,awg=1)
    for i in range(1):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield s 
