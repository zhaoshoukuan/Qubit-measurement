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
import numpy as np, sympy as sy, serial, time, datetime
from numpy.lib.function_base import iterable
from easydl import clear_output
from qulab.job import Job
from qulab.wavepoint import WAVE_FORM as WF
from collections import Iterable
from tqdm import tqdm_notebook as tqdm
from qulab import computewave_wave, optimize, dataTools
import pandas as pd
import asyncio, imp, scipy, logging
cww = imp.reload(computewave_wave)
op = imp.reload(optimize)
dt = imp.reload(dataTools)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=r'D:\skzhao\log\qubit.log', level=logging.DEBUG, format=LOG_FORMAT)

# logging.debug("This is a debug log.")
# logging.info("This is a info log.")
# logging.warning("This is a warning log.")
# logging.error("This is a error log.")
# logging.critical("This is a critical log.")

t_new = np.linspace(-15000,5000,50000)*1e-9
# t_new = np.linspace(-90000,10000,250000)*1e-9
t_list = t_new*1e9 - np.min(t_new)*1e9
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

################################################################################
# qubit
################################################################################

class qubit():
    def __init__(self,**kws):
        attribute = ['q_name','inst','T_bias','T_z','specinterp','specfunc','specfuncz','specfunc_cavity','photonnum_func','bias','zpulse','f_lo','delta','f_ex','delta_ex','alpha',\
             'power_ex','power_rabi','pi_len','pi_len2','during','T1','state','timing','envelopename','nwave','readamp','ringup','ringupamp',\
                 'weight','readmatrix','readvolt','amp','amp2','zCali',\
             'seqtype','detune','shift','phaseim','phase','phaseDiff','DRAGScaling','volt','offset','vpp']
        for j in attribute:
            self.__setattr__(j,None)
        if len(kws) != 0:
            for i in kws:
                self.__setattr__(i,kws[i])
    def asdict(self):
        return self.__dict__

    def replace(self,**kws):
        for i in kws:
            # if self.__getattribute__(i):
            if hasattr(self,i):
                self.__setattr__(i,kws[i])
            else:
                raise(f'atrribute {i} is not existed')

class common():
    global t_list, t_new
    def __init__(self,freqall,ats,dc,psg,awg,attinst,jpa,qubits={}):
    
        self.freqall = freqall
        self.ats = ats
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.jpa = jpa
        self.qubits = {i.q_name:i for i in qubits} 
        self.wave = {}
        self.attinst = attinst
        self.com_lst = [f'com{i}' for i in np.arange(3,15)]
        self.inststate = 0
        self.t_list = t_list
        self.t_new = t_new
        self.t_range = (-90e-6,10e-6)
        self.sample_rate = 2.5e9
        self.readamp = [0.08]*10
        self.ringup = [100]*10
        self.ringupamp = [0.1]*10
        self.mode = 'hbroadcast'
        self.steps = 101
        self.onwhich = {} 
        self.offwhich = {}
        self.predict = {}
        self.readmatrix = {}
        
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
# 调度器
################################################################################

async def dispatcher(measure,whichexe,processlist,paraslist,tags=[],maxlist=[500],avg=[],update_state=False):
    for j in whichexe:
        title = processlist[j].__name__
        job = Job(processlist[j], (measure,),paraslist[j], tags=tags,max=maxlist[j],avg=avg[j])
        data = await job.done()
        res_op = op.exeFit(measure,title,data,paraslist[j])
        for k in measure.qubits:
            if k in list(res_op.keys()):
                attr = res_op[k] 
                measure.qubits[k].replace(**attr)
            if update_state[j]:
                state = await cww.QueryInst(measure)
                measure.qubits[k].replace(state=[state,measure.delta,measure.base,measure.readlen])
        clear_output()
    return

################################################################################
# 打印日志
################################################################################
        
def printLog(info,mode='w'):
    with open(r'D:\skzhao\log\qubit.log', mode=mode) as filename:
        for i in info:
            filename.write(str(i)+':')
            filename.write(str(info.get(i)))
            filename.write('\n')

################################################################################
# 并行处理
################################################################################
'''
    此处代码及其啰嗦，有空修整
'''
def namepack(task):
    # task_run1 = []
    for i in task:
        keyname = list(i.keys())
        task1, task2, task3, task4 = [], [], [], []
        for j in keyname:
            if 'awg131' in j:
                task1.append(i[j])
            if 'awg132' in j:
                task2.append(i[j])
            if 'awg133' in j:
                task3.append(i[j])
            if 'awg134' in j:
                task4.append(i[j])
#         print(task1, task2, task3, task4)
        for k in range(8):
            task_run1 = []
            if len(task1) == len(task2) == len(task3) == len(task4) == 0:
                continue
            if len(task1) != 0:
                task_run1.append(task1.pop(0))
            if len(task2) != 0:
                task_run1.append(task2.pop(0))
            if len(task3) != 0:
                task_run1.append(task3.pop(0))
            if len(task4) != 0:
                task_run1.append(task4.pop(0))
#             print(task1, task2, task3, task4)
            yield task_run1

        # task_run1.append((task1,rask2,task3,task4))
    


async def main(task):
    # for i in task:
    #     task_run = asyncio.create_task(i)
    #     await task_run
    await asyncio.gather(*task)
            
def concurrence(task):
    # asyncio.run(main([cww.openandcloseAwg(measure,'OFF')])) 
    # await cww.openandcloseAwg('ON')
    f = namepack(task)
    
    for i in f:
        if i == []:
            continue
        else:
            asyncio.run(main(i)) 
    # asyncio.run(main([cww.openandcloseAwg(measure,'ON')])) 
    # loop = asyncio.get_event_loop()
    # for i in task:
    #     if i == []:
    #         continue
    #     else:
    #         # print('loop')
    #         # # loop = asyncio.get_event_loop()
    #         # try:
    #         loop.run_until_complete(asyncio.wait(i))
    #         # except Exception as e:
    #         #     pass
    #         #     print(e)
    #         #     # 引发其他异常后，停止loop循环
    #         #     lp.stop()
    #         # finally:
    #         #     # 不管是什么异常，最终都要close掉loop循环
    #         #     loop.close()
    # # loop.close()

################################################################################
# 获取数据
################################################################################

def getSamplesPerRecode(numOfPoints):
    samplesPerRecord = (numOfPoints // 64 - 1) * 64
    return samplesPerRecord

def getExpArray(f_list, numOfPoints, weight=None, sampleRate=1e9):
    samplesPerRecord = getSamplesPerRecode(numOfPoints)
    t = np.arange(0, samplesPerRecord, 1) / sampleRate
    e = []
    for i,f in enumerate(f_list):
        weight_m = np.asarray(weight[i])
        if weight_m is None:
            weight_m = np.ones(samplesPerRecord)
        e.append(weight_m * np.exp(-1j * 2 * np.pi * f * t))
    return np.asarray(e).T

async def yieldData(measure,avg=False,fft=False,offset=True,hilbert=False,is2ch=False,filter=None):

    # chA, chB, ch_I, ch_Q = await measure.ats.getTraces(avg=avg,fft=fft,hilbert=False)
    A_lst, B_lst = await measure.ats.getData(offset)
    print(np.shape(A_lst))
    if filter is not None:
        A_lst = op.RowToRipe().smooth(A_lst,f0=filter,axis=1)
        B_lst = op.RowToRipe().smooth(B_lst,f0=filter,axis=1)
    weight = []
    for j, i in enumerate(measure.qubitToread):
        qubit = measure.qubits[i]
        weight.append(qubit.weight)
    e = getExpArray(measure.delta,measure.readlen,weight)
    n = e.shape[0]
    if hilbert:
        Analysis_cos = scipy.signal.hilbert(A_lst,axis=1)
        Analysis_sin = scipy.signal.hilbert(B_lst,axis=1)
        # theta = np.angle(Analysis_cos) - np.angle(Analysis_sin)
        # Analysis_sin *= np.exp(1j*(theta))
        # A_lst, B_lst = (np.real(Analysis_cos) + np.real(Analysis_sin)), (np.imag(Analysis_cos) + np.imag(Analysis_sin)) 
        if is2ch:
            A_lst, B_lst = (np.real(Analysis_cos) - np.imag(Analysis_sin)), (np.imag(Analysis_cos) + np.real(Analysis_sin))
        else: 
            A_lst, B_lst = np.real(Analysis_cos), np.imag(Analysis_cos)
            A_lst_Q, B_lst_Q = np.real(Analysis_sin), np.imag(Analysis_sin)
    if fft:
        print(np.shape(A_lst),np.shape(e))
        A_lst = (A_lst[:, :n]).dot(e)
        B_lst = (B_lst[:, :n]).dot(e)
        if hilbert and is2ch == False:
            A_lst_Q = (A_lst_Q[:, :n]).dot(e)
            B_lst_Q = (B_lst_Q[:, :n]).dot(e)
    if avg:
        if hilbert and is2ch == False:
            return A_lst.mean(axis=0), B_lst.mean(axis=0), A_lst_Q.mean(axis=0), B_lst_Q.mean(axis=0)
        else:
            return A_lst.mean(axis=0), B_lst.mean(axis=0), np.array([0]), np.array([0])
    else:
        if hilbert and is2ch == False:
            return A_lst, B_lst, A_lst_Q, B_lst_Q
        else:
            return A_lst, B_lst, np.array([0]), np.array([0])
    

################################################################################
# 读出校准
################################################################################

def readPop(measure,ss):
    d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
    y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
    pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
    pop_on = np.array([pop[i] if measure.onwhich[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
    pop_off = np.array([pop[i] if measure.offwhich[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
    pop_cali = [np.array(np.mat(measure.readmatrix[j]).I*np.mat([pop_off[i],pop_on[i]]).T)[:,0] for i, j in enumerate(measure.qubitToread)]
    return pop_cali

async def seqReadpop(measure,len_data,repeats=600,hilbert=False):
    pop = []
    for i in range(repeats):
        ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=hilbert)
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        pop.append(ss[1:len_data+1,:])
    pop_m, pop = [], np.array(pop)
    for i in range(len_data):
        pop_m.append(np.array(readPop(measure,pop[:,i,:]))[0,1])
    return pop_m
    

################################################################################
# AWG恢复设置
################################################################################

async def resetAwg(awg):
    global t_new
    for i in awg:  
        await awg[i].setValue('Run Mode','Triggered')   # Triggered, Continuous
        for j in range(8):
            await cww.genwaveform(awg[i],[f'ch{j+1}'],[j+1])
            await awg[i].update_waveform(np.zeros((len(t_new),)),f'ch{j+1}')
            await awg[i].setValue('Vpp',1.5,ch=j+1)
            await awg[i].setValue('Offset',0,ch=j+1)
            await awg[i].write('OUTPUT%d:WVALUE:ANALOG:STATE %s'%(j+1,'FIRST'))  #FIRST, ZERO
            await awg[i].write('SOUR%d:TINP %s'%((j+1),'ATR'))        
            for k in range(4):
                await awg[i].write('OUTPUT%d:WVALUE:MARKER%d %s'%(j+1,(k+1),'FIRST'))            #FIRST, LOW, HIGH
        # if i == 'awg_trig':
        #     await cww.genwaveform(awg[i],['Readout_Q'],[5])
        #     await cww.couldRun(measure,awg[i])
        #     await awg[i].write('TRIGg:SOUR %s'%'INT')   
        #     await awg[i].write('TRIGGER:INTERVAL %f'%260e-6) 
        #     for m in range(8):
        #         await awg[i].write('SOUR%d:TINP %s'%((m+1),'ITR'))

################################################################################
# AWG同步
################################################################################

async def awgSync(measure):
    t_list = measure.t_list
    wf = WF(t_list)
    awg_list = measure.awg
    for i in awg_list:
        wait = 'ATR' if awg_list[i] == awg_list['awg_trig'] else 'ATR'
        await awg_list[i].create_waveform(name=''.join((i,'_sync1')),length=len(measure.t_list),format=None)
        await awg_list[i].create_sequence(name=''.join((i,'_syncSeq')),steps=3,tracks=1)
        height,t_end,width = [1],[np.max(t_list)/2],[2000]
        sample = wf.square_wave(t_end,width,height)
        await awg_list[i].update_marker(name=''.join((i,'_sync1')),mk1=sample)
        for j in range(3):
            j += 1
            goto = 'FIRST' if j == 3 else 'NEXT'
            await awg_list[i].set_sequence_step(name=''.join((i,'_syncSeq')),sub_name=[''.join((i,'_sync1'))],\
                                                   step=j,wait=wait,goto=goto,repeat=1,jump=None)
        await awg_list[i].use_sequence(name=''.join((i,'_syncSeq')),channels=[8])
        await awg_list[i].query('*OPC?')
        await awg_list[i].output_on(ch=8)
        await awg_list[i].setValue('Force Jump',1,ch=8)
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
    # f_lo = f_cavity.min() - 20e6 
    f_lo = f_cavity.max() + 20e6
    if len(f_cavity) == 1:
        f_lo = f_cavity.mean()+20e6
    delta =  f_lo - f_cavity 
    n = len(f_cavity)
    return f_lo, delta, n

################################################################################
# 直流源设置
################################################################################

async def dcManage(measure,dcstate={},readstate=[],calimatrix=None,qnum=10,readamp=0.3):
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
        await cww.modulation_read(measure,delta,readlen=measure.readlen)
        await cww.couldRun(measure,measure.awg['awgread'])
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

async def exManage(measure,exstate=[],qnum=10):
    qubits = measure.qubits
    if len(exstate) == 1:
        q_target = qubits[exstate[0]]
        ex_lo = q_target.f_ex + q_target.delta_ex
        delta_ex = {exstate[0]:q_target.delta_ex}
        await measure.psg[q_target.inst['ex_lo']].setValue('Frequency',ex_lo)
        await measure.psg[q_target.inst['ex_lo']].setValue('Output','ON')
    else:
        f_ex1, f_ex2, f_ex3 = {}, {}, {}
        delta_ex1, delta_ex2, delta_ex3 = {}, {}, {}
        for i,j in enumerate(qubits):
            if j in exstate:
                if qubits[j].inst['ex_lo'] == 'psg_ex1':
                    f_ex1[j] = qubits[j].f_ex
                if qubits[j].inst['ex_lo'] == 'psg_ex2':
                    f_ex2[j] = qubits[j].f_ex
                if qubits[j].inst['ex_lo'] == 'psg_ex3':
                    f_ex3[j] = qubits[j].f_ex
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

    return delta_ex

async def zManage(measure,dcstate={},calimatrix=None,qnum=10):
    len_data = 0
    for i,j in enumerate(dcstate):
        x = dcstate[j]
        if not isinstance(dcstate[j],Iterable):
            x = np.array([dcstate[j]])
        dcstate[j] = x
        len_data = len(x)
            
    bias_dict = {}
    qubits = measure.qubits
    if qnum < 10:
        qubits = {}
        for i in measure.qubits:
            if i in dcstate:
                qubits[i] = measure.qubits[i]
    for i,j in enumerate(qubits):

        bias_dict[j] = np.array(dcstate[j]) if j in dcstate else np.array([0]*len_data)

    bias = np.array(list(bias_dict.values()))

    matrix = np.mat(np.eye(qnum)) if calimatrix is None else np.mat(calimatrix)
        
    current = np.array(matrix.I * np.mat(bias))
    current = {j:current[i,:] for i,j in enumerate(qubits)}
    return current

################################################################################
# 执行激励seq并行处理
################################################################################

async def executeEXseq(measure,update_seq,len_data,comwave,exstate=[],readseq=True,mode='vbroadcast',**paras):
    delta_ex = await exManage(measure,exstate=exstate)
    print(delta_ex)
    bit = measure.qubits
    task1, task2, task3, namelist, awglist = {}, {}, {}, [], []
    for i in delta_ex:
        qubit = bit[i]
        qubit.delta_ex = delta_ex[i]
        taskname = ''.join((qubit.inst['ex_awg'],'_ch',str(qubit.inst['ex_ch'][0])))
        # kind = ''.join((qubit.inst['ex_awg'],qubit.q_name,'_ex'))
        kind = ''.join((qubit.q_name,'_ex'))
        awg = measure.awg[qubit.inst['ex_awg']]
        # namelist.append(kind)
        # awglist.append(awg)
        await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data,readseq,mode)
        if comwave:
            task_ready = update_seq(qubit,measure,kind=kind,**paras)
            # task1.append(task_ready)
            task1[taskname] = task_ready
        task_manageawg = cww.awgchmanage(measure,awg,kind,qubit.inst['ex_ch'])
        # task2.append(task_manageawg)
        task2[taskname] = task_manageawg
        if readseq:
            taskread = ''.join(('awg133','_read'))
            measure.wave['Read'] = [['Readout_I']*len_data,['Readout_Q']*len_data,['Read_sub']*len_data]
            # await measure.awg['awgread'].create_sequence('Read_sub',2,2)
            await cww.genSeq(measure,measure.awg['awgread'],'Read',mode=mode)
            read_manageawg = cww.awgchmanage(measure,measure.awg['awgread'],'Read',[1,5])
            task2[taskread] = read_manageawg
            read_run = cww.couldRun(measure,measure.awg['awgread'])
            task3[taskread] = read_run
        task_run = cww.couldRun(measure,awg)
        task3[taskname] = task_run
    return [task1,task2,task3]

################################################################################
# 执行zseq并行处理
################################################################################

async def executeZseq(measure,update_seq,len_data,comwave,dcstate={},calimatrix=None,readseq=True,qnum=10,mode='vbroadcast',**paras):
    # current = await zManage(measure,dcstate,calimatrix=calimatrix,qnum=qnum)
    bit = measure.qubits
    task1, task2, task3, namelist, awglist = {}, {}, {}, [], []
    for i in dcstate:
        qubit = bit[i]
        taskname = ''.join((qubit.inst['z_awg'],'_ch',str(qubit.inst['z_ch'][0])))
        # kind = ''.join((qubit.inst['z_awg'],qubit.q_name,'_z'))
        kind = ''.join((qubit.q_name,'_z'))
        awg = measure.awg[qubit.inst['z_awg']]
        await cww.gen_packSeq(measure,kind,awg,['step'],len_data,readseq,mode)
        if comwave:
            # if 'volt' in paras:
            # paras['volt'] = current[i]
            task_ready = update_seq(qubit,measure,kind=kind,**paras)
            # else:
            #     task_ready = update_seq(qubit,measure,kind,volt=current[i],**paras)
            # task1.append(task_ready)
            task1[taskname] = task_ready
        task_manageawg = cww.awgchmanage(measure,awg,kind,qubit.inst['z_ch'])
        # task2.append(task_manageawg)
        task2[taskname] = task_manageawg
        if readseq:
            taskread = ''.join(('awg133','_read'))
            measure.wave['Read'] = [['Readout_I']*len_data,['Readout_Q']*len_data,['Read_sub']*len_data]
            # await measure.awg['awgread'].create_sequence('Read_sub',2,2)
            await cww.genSeq(measure,measure.awg['awgread'],'Read',mode=mode)
            read_manageawg = cww.awgchmanage(measure,measure.awg['awgread'],'Read',[1,5])
            task2[taskread] = read_manageawg
            read_run = cww.couldRun(measure,measure.awg['awgread'])
            task3[taskread] = read_run
        task_run = cww.couldRun(measure,awg)
        task3[taskname] = task_run
    return [task1,task2,task3]
    # concurrence([task1,task2])

################################################################################
# 执行激励并行处理
################################################################################

async def executeEXwave(measure,update_wave,exstate=[],output=True,**paras):
    delta_ex = await exManage(measure,exstate=exstate)
    bit = measure.qubits
    task1, task2, namelist, awglist = {}, {}, [], []
    for i in delta_ex:
        qubit = bit[i]
        taskname = ''.join((qubit.inst['ex_awg'],'_ch',str(qubit.inst['ex_ch'][0])))
        qubit.delta_ex = delta_ex[i]
        namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
        chlist = qubit.inst['ex_ch']
        awg = measure.awg[qubit.inst['ex_awg']]
        # pulselist = await cww.funcarg(update_wave,qubit,**paras)
        pulselist = await cww.funcarg(update_wave,qubit,**paras)
        task_ready = cww.writeWave(measure,awg,name=namelist,pulse=pulselist)
        # task1.append(task_ready)
        task1[taskname] = task_ready
        if output:
            task_manageawg = cww.couldRun(measure,awg,chlist,namelist)
            # task2.append(task_manageawg)
            task2[taskname] = task_manageawg
        else:
            task_manageawg = cww.couldRun(measure,awg)
            # task2.append(task_manageawg)
            task2[taskname] = task_manageawg
    return [task1,task2]

################################################################################
# 执行z并行处理
################################################################################

async def executeZwave(measure,update_wave,dcstate={},qnum=10,calimatrix=None,output=True,args='volt',**paras):
    current = await zManage(measure,dcstate=dcstate,calimatrix=calimatrix,qnum=qnum)
    # print(current)
    bit = measure.qubits
    task1, task2, awglist = {}, {}, []
    for i in current:
        qubit = bit[i]
        # print(qubit.q_name)
        taskname = ''.join((qubit.inst['z_awg'],'_ch',str(qubit.inst['z_ch'][0])))
        zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
        awg = measure.awg[qubit.inst['z_awg']]
        # if 'volt' in paras:
        #     pulselist = await cww.funcarg(update_wave,qubit,**paras)
        # else:
        paras[args] = current[i]
        pulselist = await cww.funcarg(update_wave,qubit,**paras)
        task_ready = cww.writeWave(measure,awg=awg,name=zname,pulse=pulselist)
        # task1.append(task_ready)
        task1[taskname] = task_ready
        if output:
            task_manageawg = cww.couldRun(measure,awg,zch,zname)
            # task2.append(task_manageawg)
            task2[taskname] = task_manageawg
        else:
            task_manageawg = cww.couldRun(measure,awg)
            # task2.append(task_manageawg)
            task2[taskname] = task_manageawg
    return [task1,task2]

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
# S21
################################################################################

async def S21(qubit,measure,modulation=False,f_lo=None,f=None):
    #await jpa_switch(measure,state='OFF')
    if f_lo == None:
        f_lo, delta, n = await resn(np.array(qubit.f_lo))
        freq = np.linspace(-3,3,81)*1e6 + f_lo
    else:
        freq = np.linspace(-3,3,81)*1e6 + f_lo
        delta, n = measure.delta, measure.n
    if modulation:
        await cww.modulation_read(measure,delta,readlen=measure.readlen)
    await cww.couldRun(measure,measure.awg['awgread'])
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
        await cww.modulation_read(measure,measure.delta,readlen=1200,phase=i)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s

################################################################################
# 优化读出IQMixer
################################################################################

async def readIQMixer(measure,var1,var2):
    optkind = 'LO'
    awg = measure.awg['awgread']
    ch_I, ch_Q = 1,5
    f_list = np.array(await measure.ats.getValue('f_list'))
    if optkind == 'LO':
        await awg.setValue('Offset',var1,ch=ch_I)
        await awg.setValue('Offset',var2,ch=ch_Q)
        time.sleep(1)
        await cww.couldRun(awg)
        I, Q = 0, 0
        for i in range(10):
            chA, chB, ch_I, ch_Q = await measure.ats.getTraces(hilbert=False,is2ch=False,offset=False)
            I += chA
            Q += chB
        I, Q = I / 10, Q / 10
    if optkind == 'Imag':
        pulse = await cww.rabiWave(envelopename=['square',1],nwave=1,amp=1,phaseDiff=var2,pi_len=2000e-9,shift=-2000e-9,delta_ex=f_list[0])
        wav_I, wav_Q, mrk1, mrk2 = pulse
        await cww.writeWave(awg,['opt_I','opt_Q'],(wav_I, wav_Q, mrk1, mrk2))
        await cww.couldRun(awg)
        I, Q = 0, 0
        for i in range(10):
            chA, chB, ch_I, ch_Q = await measure.ats.getTraces(hilbert=False,is2ch=False,offset=False)
            I += chA
            Q += chB
        I, Q = I / 10, Q / 10
    f = np.fft.fftshift(np.fft.fftfreq(len(I)))*1e9
    Pxx = np.abs(np.fft.fftshift(np.fft.fft(I + 1j*Q)))
    return  dt.nearest(f,0,np.abs(Pxx)) if optkind == 'LO' else dt.nearest(f,-f_list[0],np.abs(Pxx))

################################################################################
# 重新混频
################################################################################

async def again(qubit,measure,modulation=False,flo=None,freq=None):
    #f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    #freq = np.linspace(-2.5,2.5,121)*1e6+f_lo
    for i in measure.psg:
        if i != 'psg_lo' and i != 'psg_pump':
            await measure.psg[i].setValue('Output','OFF')
    length = len(freq) if freq is not None else 121
    job = Job(S21, (qubit,measure,modulation,flo,freq),auto_save=True,max=length,tags=[qubit.q_name])
    f_s21, s_s21 = await job.done()
    index = np.abs(s_s21).argmin(axis=0)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n = await resn(np.array(f_res))
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        #await cww.ats_setup(measure.ats,delta)
        await cww.modulation_read(measure,delta,readlen=measure.readlen)
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
    zname = [''.join((qubit.q_name,'_z'))]
    await cww.genwaveform(awg,zname,qubit.inst['z_ch'])
    await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    for i in current:
        pulse = await cww.funcarg(cww.zWave,qubit,during=2000e-9,offset=i,volt=0,shift=3000e-9)
        await cww.writeWave(measure,awg,zname,pulse,mark=False)
        await cww.couldRun(measure,awg)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# S21vsPower
################################################################################

async def S21vsPower(qubit,measure,att,com='com8'):
    l = measure.readlen
    await dcManage(measure,dcstate={},readstate=[qubit.q_name],calimatrix=None)
    for i in att:
        # await measure.attinst['com8'].set_att(i)
        measure.readamp[0] = i
        await cww.modulation_read(measure,measure.delta,readlen=2000)
        job = Job(S21, (qubit,measure,False,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21
    measure.readlen = l

################################################################################
# SingleSpec
################################################################################

async def singlespec(measure,freq,modulation=False,f_read=None,readponit=True,exstate=[]):
    qubit = measure.qubits[exstate[0]]
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
    # await measure.psg['psg_trans'].setValue('Frequency',ftarget)
    # f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,measure.f_lo)
    # await measure.psg['psg_trans'].setValue('Output','ON')
    for i in bias:
        i -= np.mean(bias)
        zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
        z_awg = measure.awg[qubit.inst['z_awg']]
        pulse = await cww.funcarg(cww.zWave,qubit,during=500/1e9,volt=i,shift=500e-9,readvolt=0)
        await cww.writeWave(measure,z_awg,zname,pulse)
        await cww.couldRun(measure,z_awg,zch,zname)
        # await measure.dc[qubit.q_name].DC(i)
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
# Spec2d
################################################################################

async def spec2d(qubit,measure,freq,calimatrix,modulation=False):
    current = np.linspace(-qubit.T_bias[0]*0.31,qubit.T_bias[0]*0.31,32) + qubit.T_bias[1] 
    # await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    await dcManage(measure,dcstate={},readstate=[qubit.q_name],calimatrix=calimatrix)
    for i in current:
        await dcManage(measure,dcstate={qubit.q_name:i},readstate=None,calimatrix=calimatrix)
        # await measure.dc[qubit.q_name].DC(i)
        job = Job(singlespec, (qubit,measure,freq,modulation,measure.f_lo,True),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

################################################################################
# Spec2d_awg
################################################################################

async def spec2d_awg(qubit,measure,current,freq,calimatrix,modulation=False):
    
    # namelist = [f'ch{i}' for i in qubit.inst['z_ch']]
    # chlist = qubit.inst['z_ch']
    # z_awg = measure.awg[qubit.inst['z_awg']]
    task = await executeZwave(measure,cww.zWave,dcstate={},\
            calimatrix=calimatrix,during=0/1e9,shift=100e-9)
    concurrence(task)
    # dcstate = {i: round(measure.qubits[i].T_bias[1]+measure.qubits[i].T_bias[0]/2,3) for i in measure.qubits}
    # dcstate[qubit.q_name] = round(qubit.T_bias[1],3)
    # dcstate = {i: measure.qubits[i].T_bias[1] for i in measure.qubits }
    # await dcManage(measure,dcstate=dcstate,readstate=[],calimatrix=None)
    res = await again(qubit,measure,False,measure.f_lo)
    
    
    for j,i in enumerate(current):
        output = True if j== 0 else False
        task = await executeZwave(measure,cww.zWave,dcstate={qubit.q_name:i},\
            calimatrix=calimatrix,output=output,during=(len(measure.t_new)/2.5/2e9+2000e-9),shift=200e-9)
        concurrence(task)
        # flux = await zManage(measure,dcstate={qubit.q_name:i},calimatrix=calimatrix)
        # pulselist = await cww.funcarg(cww.zWave,qubit,pi_len=27000e-9,volt=flux[qubit.q_name],shift=100e-9)
        # # pulselist = await cww.funcarg(cww.rabiWave,qubit,pi_len=27000e-9,amp=flux[qubit.q_name])
        # await cww.writeWave(measure,z_awg,name=namelist,pulse=pulselist[:1])
        # await cww.couldRun(measure,z_awg,chlist,namelist)
        job = Job(singlespec, (measure,freq,modulation,measure.f_lo,False,[qubit.q_name]),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

################################################################################
# Rabi
################################################################################

async def rabi(measure,amp,arg='amp',exstate=[]):
    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    # qubit = measure.qubits[exstate[0]]
    # namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    # chlist = qubit.inst['ex_ch']
    # awg_ex = measure.awg[qubit.inst['ex_awg']]
    # await cww.couldRun(measure,awg_ex,chlist,namelist)

    await measure.psg['psg_lo'].setValue('Output','ON')
    
    for j,i in enumerate(amp):
        para = {arg:i} if arg == 'amp' else {arg:i/1e9}
        output = True if j == 0 else False
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output=output,**para)
        concurrence(task)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        n = np.shape(s)[-1]
        yield [i]*n, s-measure.base

################################################################################
# Rabi_seq
################################################################################

async def rabi_seq(measure,amp,arg='amp',exstate=[],comwave=True,readseq=True,mode='vbroadcast'):
    # await cww.openandcloseAwg(measure,'OFF')
    x = amp if arg == 'amp' else amp/1e9
    len_data, t = len(amp), np.array([amp]*measure.n).T
    task = await executeEXseq(measure,cww.Rabi_sequence,len_data,comwave,exstate,readseq,mode,v_or_t=x,arg=arg)
    concurrence(task)
    # await cww.openandcloseAwg(measure,'ON')

    await measure.psg['psg_lo'].setValue('Output','ON')
    x = amp if mode == 'hbroadcast' else t

    if mode == 'hbroadcast':
        for i in x:
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            # time.sleep(0.5)
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            n = np.shape(s)[-1]
            yield [i]*n, s - measure.base
    if mode == 'vbroadcast':
        for i in range(300):
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            yield x, s - measure.base

################################################################################
# Rabi2_seq
################################################################################

async def rabi2_seq(measure,amp,arg='amp',exstate=[],comwave=True,readseq=True,mode='vbroadcast'):
    # await cww.openandcloseAwg(measure,'OFF')
    x = amp if arg == 'amp' else amp/1e9
    len_data, t = len(amp), np.array([amp]*measure.n).T
    task = await executeEXseq(measure,cww.Rabi2_sequence,len_data,comwave,exstate,readseq,mode,v_or_t=x,arg=arg)
    concurrence(task)
    # await cww.openandcloseAwg(measure,'ON')

    await measure.psg['psg_lo'].setValue('Output','ON')
    x = amp if mode == 'hbroadcast' else t

    if mode == 'hbroadcast':
        for i in x:
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            n = np.shape(s)[-1]
            yield [i]*n, s - measure.base
    if mode == 'vbroadcast':
        for i in range(300):
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            yield x, s - measure.base

################################################################################
# 优化pi脉冲
################################################################################
    
async def pipulseOpt(measure,nwave,wavlen,optwhich='pi_len',exstate=[],mode='vbroadcast'):
    qubit = measure.qubits[exstate[0]]
    if optwhich == 'pi_len':
        pilen = qubit.pi_len
        start = 0.5*pilen if 0.5*pilen > 10 else 1
        end = 2.5*pilen if 0.5*pilen > 10 else pilen+30
        x = np.linspace(start,end,wavlen)
        # func = rabiTime_seq
    if optwhich == 'amp':
        amp = qubit.amp
        start = 0.5*amp
        end = 1.5*amp if 1.5*amp <=1 else 1
        x = np.linspace(start,end,wavlen)
        # func = rabiPower_seq
    for j,i in enumerate(range(nwave)):
        readseq = True if j == 0 else False
        qubit.nwave = 4*i+1
        numrepeat, avg = (len(x),False) if mode == 'hbroadcast' else (300,True)
        job = Job(rabi_seq, (measure,x,optwhich,exstate,True,readseq,mode),max=numrepeat,avg=avg,auto_save=False)
        t_r, s_r = await job.done()

        # job = Job(rabi_many, (measure,x,optwhich,exstate), max=300,avg=True,auto_save=False)
        # t_r, s_r = await job.done()
        yield [4*i+1]*measure.n, t_r, s_r
    qubit.nwave = 1

################################################################################
# IQ-Mixer 线性度
################################################################################
    
async def lineIQMixer(measure,amp,t_rabi,exstate=[],mode='vbroadcast'):
    qubit = measure.qubits[exstate[0]]
    for j,i in enumerate(amp):
        readseq = True if j == 0 else False
        qubit.amp = i
        numrepeat, avg = (len(t_rabi),False) if mode == 'hbroadcast' else (300,True)
        job = Job(rabi_seq, (measure,t_rabi,'pi_len',exstate,True,readseq,mode),max=numrepeat,avg=avg,auto_save=False)
        t_r, s_r = await job.done()

        # job = Job(rabi_many, (measure,x,optwhich,exstate), max=300,avg=True,auto_save=False)
        # t_r, s_r = await job.done()
        yield [i]*measure.n, t_r, s_r
    qubit.amp = 1

################################################################################
# Ramsey_seq
################################################################################

async def Ramsey_seq(measure,t_Ramsey,exstate=[],comwave=True,readseq=True,mode='vbroadcast'):
    measure.qubits[exstate[0]].replace(nwave=0,seqtype='PDD',detune=2e6)
    len_data, t = len(t_Ramsey), np.array([t_Ramsey]*measure.n).T
    task = await executeEXseq(measure,cww.Coherence_sequence,len_data,comwave,exstate,readseq,mode,v_or_t=t_Ramsey/1e9,arg='t_run')
    concurrence(task)
    # await cww.openandcloseAwg('ON')
    await measure.psg['psg_lo'].setValue('Output','ON')
    
    x = t_Ramsey if mode == 'hbroadcast' else t
    if mode == 'hbroadcast':
        for i in x:
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            n = np.shape(s)[-1]
            yield [i]*n, s - measure.base
    if mode == 'vbroadcast':
        for i in range(300):
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            yield x, s - measure.base
    measure.qubits[exstate[0]].replace(nwave=1,seqtype='PDD',detune=2e6)
################################################################################
# SpinEcho_seq
################################################################################

async def SpinEcho_seq(measure,t_Ramsey,exstate=[],comwave=True,readseq=True,mode='vbroadcast'):

    len_data, t = len(t_Ramsey), np.array([t_Ramsey]*measure.n).T
    task = await executeEXseq(measure,cww.Coherence_sequence,len_data,comwave,exstate,readseq,mode,v_or_t=t_Ramsey/1e9,arg='t_run')
    concurrence(task)
    await measure.psg['psg_lo'].setValue('Output','ON')
    
    for i in range(300):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# 优化读出点
################################################################################

async def readOp(measure,exstate=[]):

    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate)
    concurrence(task)

    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    for j, i in enumerate(['OFF','ON']):
        for k in exstate:
            await measure.psg[measure.qubits[k].inst['ex_lo']].setValue('Output',i)
        job = Job(S21, (qubit,measure,False,measure.f_lo),auto_save=False,max=81)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [j]*n, f_s21, s_s21

################################################################################
# 优化读出点占比
################################################################################

async def readOpweight(measure,exstate=[]):

    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate)
    concurrence(task)

    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    for j, i in enumerate(['OFF','ON']):
        for k in exstate:
            await measure.psg[measure.qubits[k].inst['ex_lo']].setValue('Output',i)
        chA, chB, ch_I, ch_Q = await measure.ats.getTraces()
        yield chA + 1j*chB

################################################################################
# 对比度
################################################################################

async def visibility(n,s0,s1):
    theta = np.arange(0, 2*np.pi, 0.01)
    data = []
    for i in range(n):
        c0, c1 = np.mean(s0), np.mean(s1)
        s0 = s0 / ((c1-c0)/np.abs(c1-c0))
        s1 = s1 / ((c1-c0)/np.abs(c1-c0))
        s0 = np.real(s0)
        s1 = np.real(s1)
        bins = np.linspace(np.min(np.r_[s0,s1]), np.max(np.r_[s0,s1]), 61)
        y0,_ = np.histogram(s0, bins=bins)
        y1,_ = np.histogram(s1, bins=bins)
        inte0 = np.cumsum(y0)/np.sum(y0)
        inte1 = np.cumsum(y1)/np.sum(y0)
        inte_diff = np.cumsum(y0)/np.sum(y0) - np.cumsum(y1)/np.sum(y1)
        offstd, onstd = np.std(s0), np.std(s1)
        roff = np.real(c0) + offstd * np.cos(theta)
        ioff = np.imag(c0) + offstd * np.sin(theta)
        ron = np.real(c1) + onstd * np.cos(theta)
        ion = np.imag(c1) + onstd * np.sin(theta)
        data.append([inte0,inte1,inte_diff,(roff,ioff),(ron,ion)])
    return data

################################################################################
# 临界判断
################################################################################

async def threshHold(measure,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    ex_name, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate)
    concurrence(task)
    await cww.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=3000)
    for j, i in enumerate(['OFF','ON','ON2']):
        if i != 'ON2':
            await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
            time.sleep(0.1)
        # else:
        #     pulse_rabi2 = await cww.funcarg(cww.singleQgate,qubit,pi_len=qubit.pi_len2,amp=qubit.amp2,\
        #         delta_ex=(qubit.delta_ex + qubit.alpha))
        #     pulse_pi1 = await cww.funcarg(cww.singleQgate,qubit,shift=qubit.envelopename[1]*qubit.pi_len2 + 10e-9)
        #     pulse = np.array(pulse_pi1) + np.array(pulse_rabi2)
        #     await cww.writeWave(measure,ex_awg,ex_name,pulse,mark=False)
        #     await cww.couldRun(measure,ex_awg,exch)
        
        ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=False)
        # ch_A, ch_B, I, Q = await yieldData(measure,avg=False,fft=True,hilbert=False,filter=None)
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        # s = Am + Bm
        s = Am + 1j*Bm
        sq = I + 1j*Q
        yield j, s, sq

################################################################################
# 优化JPA
################################################################################

async def singleJpa(measure,pumppower,exstate=[]):
    for j in pumppower:
        await measure.psg['psg_pump'].setValue('Power',j)
        job = Job(threshHold, (measure,exstate), tags=exstate,no_bar=True,auto_save=False)
        st, s_st, s_st_Q = await job.done()
        s_off, s_on = s_st[0,:,0], s_st[1,:,0]
        mean_off, mean_on = np.mean(s_off), np.mean(s_on)
        d = np.abs(mean_on-mean_off)
        std = (np.std(s_off)+np.std(s_on))/2
        yield [j], [d/std]
async def optJpa(measure,current,pumppower,exstate=[]):
    for i in current:
        await measure.dc['jpa'].DC(i)
        job = Job(singleJpa, (measure,pumppower,exstate), tags=exstate,max=len(pumppower))
        pump,snr = await job.done()
        yield [i],pump,snr

################################################################################
# 优化读出功率
################################################################################

async def readpowerOpt(measure,which,readamp):
    # qubit = measure.qubits[exstate[0]]
    n = len(measure.delta)
    for k in readamp:
        measure.readamp = [k]
        await cww.modulation_read(measure,measure.delta,readlen=measure.readlen,repeats=5000)
        await cww.couldRun(measure,measure.awg['awgread'])
        
        ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=False)
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        # s = Am + Bm
        ss = Am + 1j*Bm
        d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(n)]
        y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
        pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
        pop = np.array([pop[i] if j == 1 else 1-pop[i] for i, j in enumerate(which)])
        # d = list(zip(np.real(ss[:,0]),np.imag(ss[:,0])))
        # y = measure.predict[measure.qubitToread[0]](d)
        # pop = list(y).count(which[0])/len(y)
        yield [k]*n, pop


        # S = list(s_off) + list(s_on)
        # x,z = np.real(S), np.imag(S)
        # d = list(zip(x,z))
        # y = measure.predict[qubit.q_name](d)
        # pop = list(y).count(1)/len(y)
        # yield [i]*n, [pop]

    #     offmean, onmean = np.mean(x), np.mean(y)
    #     offstd, onstd = np.std(x), np.std(y)
    #     # doff, don = np.abs(x-offmean), np.abs(y-onmean)
    #     d = np.abs(offmean-onmean)
    #     # popoff, popon = d/offstd, d/onstd
    #     snr = 2*d**2/(offstd+onstd)**2
    #     # popoff, popon = list(doff<offstd).count(True)/len(x), list(don<onstd).count(True)/len(y)
    #     # yield [i]*n, [popoff]*n, [popon]*n
    #     yield [i]*n, [snr]*n
    # att_setup.close()

################################################################################
# AllXY drag detune
################################################################################
    
async def AllXYdragdetune(measure,which,exstate=[]):
    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    coef = np.linspace(-2,2,41)

    qubit = measure.qubits[exstate[0]]
    alpha = qubit.alpha*2*np.pi
    ex_name = [f'ch{i}' for i in qubit.inst['ex_ch']]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]

    # await cw.modulation_read(measure,measure.delta,readlen=measure.readlen)
    # await cw.genwaveform(ex_awg,qname,qubit.inst['ex_ch'])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)
    for j in [['X','Yhalf'],['Y','Xhalf']]:
        for i in coef:
            pulse1 = await cww.funcarg(cww.singleQgate,qubit,axis=j[0],DRAGScaling=i/alpha,shift=(qubit.pi_len+5e-9))
            pulse2 = await cww.funcarg(cww.singleQgate,qubit,axis=j[1],DRAGScaling=i/alpha)
            pulse = np.array(pulse1) + np.array(pulse2)
            await cww.writeWave(measure,ex_awg,ex_name,pulse)
            await cww.couldRun(measure,ex_awg)

            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            yield [i], s-measure.base

################################################################################
# Opt pi detune
################################################################################

async def optPidetune(measure,exstate=[]):
    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    qubit = measure.qubits[exstate[0]]
    deltaf = np.linspace(-10,10,101)*1e6 + qubit.f_ex

    ex_name = [f'ch{i}' for i in qubit.inst['ex_ch']]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    pulse1 = await cww.funcarg(cww.singleQgate,qubit,axis='X',shift=(3*qubit.envelopename[1]*qubit.pi_len+6e-9))
    pulse2 = await cww.funcarg(cww.singleQgate,qubit,axis='Xn',shift=(2*qubit.envelopename[1]*qubit.pi_len+4e-9))
    pulse3 = await cww.funcarg(cww.singleQgate,qubit,axis='X',shift=(1*qubit.envelopename[1]*qubit.pi_len+2e-9))
    pulse4 = await cww.funcarg(cww.singleQgate,qubit,axis='Xn',shift=(0e-9))
    pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse3) + np.array(pulse4)
    await cww.writeWave(measure,ex_awg,ex_name,pulse)
    # await cw.modulation_read(measure,measure.delta,readlen=measure.readlen)
    # await cw.genwaveform(ex_awg,qname,qubit.inst['ex_ch'])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)
    for j in deltaf:
        await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',(j+qubit.delta_ex))
        ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=False)
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        pop = readPop(measure,ss)
        yield [j], np.array(pop)[:,0] 

################################################################################
# T1_seq
################################################################################

async def T1_seq(measure,t_T1,exstate=[],comwave=True):

    len_data, t = len(t_T1), np.array([t_T1]*measure.n).T
    task = await executeEXseq(measure,cww.Rabi_sequence,len_data,comwave,exstate=exstate,v_or_t=t_T1/1e9,arg='shift')
    concurrence(task)

    await measure.psg['psg_lo'].setValue('Output','ON')
    
    for i in range(300):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base


################################################################################
# AC-Stark（XY与Readout时序矫正)
################################################################################

async def singleacStark(measure,t_shift,power=1,exstate=[],comwave=True):

    pulselist = await cww.ac_stark_wave(measure,power)
    await cww.writeWave(measure,measure.awg['awgread'],['Readout_I','Readout_Q'],pulselist,False,mark=False)
    job = Job(rabi_seq, (measure,t_shift,'shift',exstate,comwave,True,'vbroadcast'), max=300,avg=True)
    # job = Job(T1_seq, (measure,t_shift,exstate,comwave), tags=(exstate+['acstark']), max=300,avg=True)
    t_t, s_t = await job.done()
    yield t_t, s_t

async def acStark(measure,t_T1,power,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    freq = np.linspace(-260,40,51)*1e6 + qubit.f_ex 
    f_m = qubit.f_ex
    # await cww.ac_stark_wave(measure)
    # await cww.couldRun(measure,measure.awg['awgread'])

    for i,j in enumerate(freq):
        # await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',j)
        qubit.f_ex = j
        comwave = True if i == 0 else False
        job = Job(singleacStark, (measure,t_T1,power,exstate,comwave), tags=exstate, no_bar=True,auto_save=False)
        t_shift, s_ac = await job.done()
        yield [j]*measure.n, t_shift, s_ac
    qubit.f_ex = f_m
    
################################################################################
# 腔内光子数
################################################################################

async def singleNum(measure,power,comwave=True):
    
    len_data, t = len(power), np.array([power]*measure.n).T
    kind = 'Read'
    awg = measure.awg['awgread']
    await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data,readseq=False)
    if comwave:
        await cww.acstarkSequence(measure,kind=kind,v_or_t=power,arg='power')
    await cww.awgchmanage(measure,awg,kind,[1,5])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex1'].setValue('Output','ON')

    for i in range(300):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base
    
async def photonNum(measure,power,end,exstate=[]):

    qubit = measure.qubits[exstate[0]]
    freq = np.linspace(-260,40,51)*1e6 + qubit.f_ex 
    f_m = qubit.f_ex

    for i,j in enumerate(freq):
        qubit.f_ex = j
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output='True',shift=end/1e9)
        concurrence(task)
        comwave = True if i == 0 else False
        job = Job(singleNum, (measure,power,comwave), no_bar=True,auto_save=False,avg=True)
        t_shift, s_ac = await job.done()
        yield [j]*measure.n, t_shift, s_ac
    qubit.f_ex = f_m

################################################################################
# Z_Pulse（Z与Readout时序矫正)
################################################################################

async def singleZpulse(measure,t_shift,exstate=[],comwave=True,updatez=True,readseq=True):
    qubit = measure.qubits[exstate[0]]
    if updatez:
        task = await executeZwave(measure,cww.zWave,dcstate={exstate[0]:0.05},output=True,\
            during=((qubit.pi_len*qubit.envelopename[1])),shift=3000e-9)
        concurrence(task)

    job = Job(rabi_seq, (measure,t_shift,'shift',exstate,comwave,readseq,'vbroadcast'), max=300,avg=True)
    t_t, s_t = await job.done()
    yield t_t, s_t

async def zPulse(measure,t_T1,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    pulse1 = await cww.funcarg(cww.zWave,qubit,during=2000/1e9,volt=0.2,shift=3000e-9)
    volt = np.linspace(-0.2,0.1,40)
    for i,j in enumerate(volt):
        pulse2 = await cww.funcarg(cww.zWave,qubit,during=4000/1e9,volt=j,shift=1500e-9)
        pulse = np.array(pulse1) + np.array(pulse2)
        await cww.writeWave(measure,z_awg,zname,pulse,mark=False)
        await cww.couldRun(measure,z_awg,zch,zname)
        comwave = True if i == 0 else False
        job = Job(singleZpulse, (measure,t_T1,exstate,comwave,False), tags=exstate, no_bar=True,auto_save=False)
        t_shift, s_z = await job.done()
        yield [j]*measure.n, t_shift, s_z


# async def singleZ(measure,t_T1,which=0,exstate=[]):
#     qubit = measure.qubits[exstate[0]]
#     exname, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
#     ex_awg = measure.awg[qubit.inst['ex_awg']]
#     await cww.couldRun(measure,ex_awg,exch,exname)
#     for k in t_T1:
#         pulselist = await cww.funcarg(cww.rabiWave,qubit,shift=k/1e9)
#         await cww.writeWave(measure,ex_awg,name=exname,pulse=pulselist)
#         await cww.couldRun(measure,ex_awg)
#         ch_A, ch_B, I, Q = await measure.ats.getIQ()
#         Am, Bm = ch_A,ch_B
#         # theta0 = np.angle(Am) - np.angle(Bm)
#         # Bm *= np.exp(1j*theta0)
#         ss = Am + 1j*Bm
#         # d = list(zip(np.real(ss),np.imag(ss)))
#         d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
#         y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
#         pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
#         pop = np.array([pop[i] if j == 1 else 1-pop[i] for i, j in enumerate(which)])
#         yield [k], pop

# async def zPulse_pop(measure,t_T1,which=0,exstate=[]):

#     qubit = measure.qubits[exstate[0]]
#     zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
#     z_awg = measure.awg[qubit.inst['z_awg']]
#     pulse1 = await cww.funcarg(cww.zWave,qubit,pi_len=2000/1e9,volt=0.8,shift=3000e-9)
#     await cww.couldRun(measure,z_awg,zch,zname)

#     volt = np.linspace(-0.21,-0.01,51)
#     await cww.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=1000)
#     for j in volt:
#         pulse2 = await cww.funcarg(cww.zWave,qubit,pi_len=4000/1e9,volt=j,shift=1500e-9)
#         pulse = np.array(pulse1) + np.array(pulse2)
#         await cww.writeWave(measure,z_awg,zname,pulse,mark=False)
#         await cww.couldRun(measure,z_awg)
#         job = Job(singleZ,(measure,t_T1,which,exstate),max=len(t_T1))
#         t, pop = await job.done()
#         yield [j], t, pop

# async def singleZ(measure,volt,height=0,exstate=[]):
#     qubit = measure.qubits[exstate[0]]
#     zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
#     z_awg = measure.awg[qubit.inst['z_awg']]
#     pulse1 = await cww.funcarg(cww.zWave,qubit,during=2000/1e9,volt=height,shift=5000e-9,readvolt=0)
#     pulse3 = await cww.funcarg(cww.zWave,qubit,during=0,volt=0,shift=0)
#     await cww.couldRun(measure,z_awg,zch,zname)
    
#     for k in volt:
#         pulse2 = await cww.funcarg(cww.zWave,qubit,during=7500/1e9,volt=k,shift=1500e-9,readvolt=0)
#         # pulse = np.array(pulse1) + np.array(pulse2)
#         pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse3)
#         await cww.writeWave(measure,z_awg,zname,pulse)
#         await cww.couldRun(measure,z_awg)
        
#         ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=False)
#         Am, Bm = ch_A,ch_B
#         # theta0 = np.angle(Am) - np.angle(Bm)
#         # Bm *= np.exp(1j*theta0)
#         ss = Am + 1j*Bm
#         # d = list(zip(np.real(ss),np.imag(ss)))
#         # d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
#         # y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
#         # pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
#         # pop = np.array([pop[i] if which[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
#         pop = readPop(measure,ss)
#         yield [k], np.array(pop)[:,1]

# async def zPulse_pop(measure,t_T1,height=0,exstate=[]):
#     qubit = measure.qubits[exstate[0]]
    
#     printLog(qubit.asdict())

#     exname, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
#     ex_awg = measure.awg[qubit.inst['ex_awg']]
#     await cww.couldRun(measure,ex_awg,exch,exname)
#     # func = lambda t: -(scipy.special.erf((t-5000)/2)+1)/2*0.8
#     sampleStep = 0.0008
#     volt_init = np.arange(-0.04,0.04,sampleStep)
#     length = len(volt_init)
#     # index = length//2
#     # index = np.argmin(np.abs(volt_init-0)) if t_T1[0] == np.min(t_T1) else np.argmin(np.abs(volt_init+0.8))
#     index = np.argmin(np.abs(volt_init-0))
#     start, end = volt_init[index]-40*sampleStep, volt_init[index]+40*sampleStep
#     volt = np.arange(start,end,sampleStep)
    
    
#     for i, j in enumerate(t_T1):
#         print(index)
#         #res = np.zeros((length,1))
#         pop_m = np.zeros((length,1))
#         #res[int(index-(len(volt)-1)/2):int(index+(len(volt)-1)/2+1),0] = volt
#         pulselist = await cww.funcarg(cww.rabiWave,qubit,shift=j/1e9)
#         await cww.writeWave(measure,ex_awg,name=exname,pulse=pulselist)
#         await cww.couldRun(measure,ex_awg)
#         job = Job(singleZ,(measure,volt,height,exstate),max=len(volt))
#         v, pop = await job.done()
#         if len(volt) % 2 == 0:
#             pop_m[int(index-len(volt)//2):int(index+len(volt)//2),0] = pop[:,0]
#         else:
#             pop_m[int(index-len(volt)//2):int(index+len(volt)//2+1),0] = pop[:,0]

#         # v = func(t_T1[i+1])
#         # index = np.argmax(pop_m[:,0])
#         # if np.max(pop_m[:,0]) > 0.7 and np.abs(volt_init[index]-v) < np.abs(np.max(volt_init)-np.min(volt_init))/45:
#         if np.max(pop_m[:,0]) > 0.7 :
#             index = np.argmax(pop_m[:,0])
#             start, end = volt_init[index]-10*sampleStep, volt_init[index]+10*sampleStep
#             if np.max(volt_init) < end:
#                 print('warning: max too little')
#                 end = np.max(volt_init)
#             if np.min(volt_init) > start:
#                 print('warning: min too big')
#                 start = np.min(volt_init)
#             volt = np.arange(start,end,sampleStep)
#         else:
#             volt = volt_init
#             index = len(volt)//2 
#             # v = func(t_T1[i+1])
#             # index = np.argmin(np.abs(volt_init-v))
#             # start, end = volt_init[index]-80*sampleStep, volt_init[index]+80*sampleStep
#             # volt = np.arange(start,end,sampleStep)
#             if np.min(volt)<np.min(volt_init) or np.max(volt)>np.max(volt_init):
#                 print(f'warning: too big or too min {np.min(volt),np.max(volt)}')
#                 # break
#         # else:
#         #     index = np.argmax(pop_m[:,0])
#         # # index_v = np.argmax(pop[:,0])
#         #     start, end = volt_init[index]-10*sampleStep, volt_init[index]+10*sampleStep
#         #     if np.max(volt_init) < end:
#         #         print('warning: max too little')
#         #         end = np.max(volt_init)
#         #     if np.min(volt_init) > start:
#         #         print('warning: min too big')
#         #         start = np.min(volt_init)
#         #     volt = np.arange(start,end,sampleStep)
#         # if j >= 4950:
#         #     volt = volt_init
#         yield [5000-qubit.pi_len*qubit.envelopename[1]/2e-9-j], volt_init.reshape(length,1), pop_m

################################################################################
# Z_Pulse_XY（Z与Readout时序矫正) population
################################################################################

async def zPulse_XY(measure,tdelay,t_int,tshift,which,exstate=[]):
    
    qubit = measure.qubits[exstate[0]]
    ex_name, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    
    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    await cww.couldRun(measure,z_awg,zch,zname)

    pulse1 = await cww.funcarg(cww.singleQgate,qubit,axis='Xhalf',shift=tshift*1e-9)
    pulse2 = await cww.funcarg(cww.singleQgate,qubit,axis='Yhalf',shift=(tshift*1e-9-t_int/1e9-qubit.pi_len))
    pulse = np.array(pulse1) + np.array(pulse2)
    await cww.writeWave(measure,ex_awg,ex_name,pulse)
    await cww.couldRun(measure,ex_awg,exch,ex_name)
    
    for i in tdelay:

        pulse = await cww.funcarg(cww.zWave,qubit,during=2000/1e9,volt=0.8,shift=(tshift*1e-9+i*1e-9))
        await cww.writeWave(measure,z_awg,zname,pulse,mark=False)
        await cww.couldRun(measure,z_awg)

        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        # d = list(zip(np.real(ss),np.imag(ss)))
        # d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
        # y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
        # pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
        # pop = np.array([pop[i] if which[j] == 1 else 1-pop[i] for i, j in enumerate(which)])
        pop = readPop(measure,ss)
        yield [i], np.array(pop)[:,1] 

################################################################################
# 2维T1
################################################################################

async def T1_2d(measure,t_rabi,v_rabi,mode='vbroadcast',dcstate=[],exstate=[],imAmp=0):
    len_data = len(t_rabi)
    qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]
    printLog(qubit_z.asdict())
    printLog(qubit_ex.asdict(),mode='a')
    
    for i, j in enumerate(v_rabi):

        task_z = await executeZseq(measure,cww.Z_sequence,len_data,True,mode=mode,dcstate=dcstate,\
            v_or_t=t_rabi/1e9,arg='during',volt=j,delta_im=0e6,imAmp=imAmp)
        concurrence(task_z)
        comwave = True if i == 0 else False
        numrepeat, avg, measure.repeat = (len(v_rabi),False,500) if mode == 'hbroadcast' else (300,True,len_data)
        job = Job(rabi_seq, (measure,t_rabi,'shift',exstate,comwave,False,mode), tags=exstate, max=numrepeat,avg=avg, no_bar=True)
        t_t, s_t = await job.done()
        yield [j]*measure.n, t_t, s_t   

################################################################################
# 真空拉比
################################################################################

async def singleVrabi(measure,v_rabi,during,mode,dcstate,exstate,imAmp=0):
    len_data, t = len(v_rabi), np.array([v_rabi]*measure.n).T
    
    task_z = await executeZseq(measure,cww.Z_sequence,len_data,True,mode=mode,dcstate=dcstate,\
        v_or_t=v_rabi,arg='volt',during=during,delta_im=0e6,imAmp=imAmp)
    concurrence(task_z)
    await measure.psg['psg_lo'].setValue('Output','ON')
    
    x = v_rabi if mode == 'hbroadcast' else t

    if mode == 'hbroadcast':
        for i in x:
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            n = np.shape(s)[-1]
            yield [i]*n, s - measure.base
    if mode == 'vbroadcast':
        for i in range(300):
            ch_A, ch_B, I, Q = await measure.ats.getIQ()
            Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            yield x, s - measure.base

async def vRabi(measure,t_rabi,v_rabi,mode='vbroadcast',dcstate=[],exstate=[],imAmp=0):
    len_data = len(v_rabi)
    qubit_ex = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit_ex.inst['ex_ch']]
    chlist = qubit_ex.inst['ex_ch']
    awg_ex = measure.awg[qubit_ex.inst['ex_awg']]
    await cww.couldRun(measure,awg_ex,chlist,namelist)
    for i, j in enumerate(t_rabi):
        pulselist = await cww.funcarg(cww.rabiWave,qubit_ex,shift=j/1e9)
        await cww.writeWave(measure,awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(measure,awg_ex)
        output = True if i == 0 else False
        # task_ex = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output=output,shift=j/1e9)
        task_z = await executeZwave(measure,cww.zWave,dcstate=dcstate,output=output,pi_len=j/1e9)
        # concurrence([{**task_ex[0],**task_z[0]},{**task_ex[1],**task_z[1]}])
        concurrence(task_z)
        # comwave = True if i == 0 else False
        numrepeat, avg, measure.repeat = (len(v_rabi),False,500) if mode == 'hbroadcast' else (300,True,len_data)
        job = Job(singleVrabi, (measure,v_rabi,j/1e9,mode,exstate,exstate,imAmp), tags=exstate, max=numrepeat,avg=avg, no_bar=True)
        t_t, s_t = await job.done()
        yield [j]*measure.n, t_t, s_t    

################################################################################
# 邻近比特zPulse矫正时序
################################################################################

async def zzTiming(measure,t_delay,exstate=[],dcstate=[]):

    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    qubit_ex = measure.qubits[exstate[0]]
    exname = [f'ch{i}' for i in qubit_ex.inst['ex_ch']]
    exch = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit_ex.inst['ex_awg']]
    awg_ex_z = measure.awg[qubit_ex.inst['z_awg']]
    qubit_z = measure.qubits[dcstate[0]]
    zname = [f'ch{i}' for i in qubit_z.inst['z_ch']]
    zch = qubit.inst['z_ch']
    awg_z = measure.awg[qubit_z.inst['z_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    pulse_z = await cww.funcarg(cww.zWave,qubit,pi_len=50/1e9,shift=1000e-9)
    await cww.writeWave(measure,awg_z,zname,pulse_z)


    for i in t_delay:
        # pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_delay=i/1e9)
        # await cww.writeWave(measure,awg_ex,name=namelist,pulse=pulselist)
        # await cww.couldRun(measure,awg_ex)
        task = await executeEXwave(measure,cww.coherenceWave,exstate=exstate,t_delay=i/1e9)
        concurrence(task)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# 纵场调制
################################################################################

async def vModulation(measure,t_rabi,v_rabi,mode='vbroadcast',dcstate=[],exstate=[],calimatrix=None):
    len_data = len(t_rabi)
    # qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]
    # zname, z_awg = qubit_ex.q_name+'_z', measure.awg[qubit_ex.inst['z_awg']]
    # await cw.create_wavelist(measure,zseqname,(z_awg,['step'],len(t_rabi),len(measure.t_list)))
    # await cw.genSeq(measure,z_awg,zseqname)
    # await cw.awgchmanage(measure,z_awg,zseqname,qubit_ex.inst['z_ch'])
    dc = {f'q{i+1}':0 for i in range(10)}
    for i, j in enumerate(v_rabi):
        dc[dcstate[0]] = j
        current = await zManage(measure,dcstate=dc,calimatrix=calimatrix)
        task_zz = await executeZseq(measure,cww.Z_sequence_im,len_data,True,mode=mode,dcstate=dcstate,\
            v_or_t=t_rabi/1e9,arg='during',delta_im=120e6,imAmp=current[dcstate[0]])
        task_zex = await executeZseq(measure,cww.Z_sequence_im,len_data,True,mode=mode,dcstate=exstate,\
            v_or_t=t_rabi/1e9,arg='during',delta_im=120e6,imAmp=current[exstate[0]])
        concurrence([{**task_zz[0],**task_zex[0]},{**task_zz[1],**task_zex[1]}])
        # concurrence(task_zz)
        comwave = True if i == 0 else False
        numrepeat, avg, measure.repeat = (len(v_rabi),False,500) if mode == 'hbroadcast' else (300,True,len_data)
        job = Job(rabi_seq, (measure,t_rabi,'shift',exstate,comwave,comwave,mode), tags=exstate, max=numrepeat,avg=avg, no_bar=True)
        t_t, s_t = await job.done()
        yield [j]*measure.n, t_t, s_t   

###################################################
###################################################

async def vModulation_single(measure,t_rabi,v_rabi,dcstate=[],exstate=[],calimatrix=None,which={}):
    dc = {i:0 for i in (dcstate+exstate)}
    for i, k in enumerate(v_rabi):
        output = True if i == 0 else False
        dc[dcstate[0]] = k
        task_z = await executeZwave(measure,cww.zWave_im,dcstate=dc,qnum=len(dc),calimatrix=calimatrix,output=output,args='imAmp',\
            during=t_rabi/1e9,delta_im=120e6)
        concurrence(task_z)
        ch_A, ch_B, I, Q = await measure.ats.getIQ(hilbert=True)
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        # d = list(zip(np.real(ss),np.imag(ss)))
        d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
        y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
        pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
        pop = np.array([pop[i] if which[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
        yield [k]*measure.n, pop

async def vModulation_pop(measure,t_rabi,v_rabi,dcstate=[],exstate=[],calimatrix=None,which={}):
    
    # qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]
    # z_awg = measure.awg[qubit.inst['z_awg']]
    # ex_awg = measure.awg[qubit.inst['ex_awg']]
    # zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    # exname, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
    # await cww.couldRun(measure,ex_awg,exch,exname)
    # # await cww.couldRun(measure,z_awg,zch,zname)

    for i, j in enumerate(t_rabi):
        output = True if i == 0 else False
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output=output,shift=j/1e9)
        concurrence(task)
        job = Job(vModulation_single, (measure,j,v_rabi,dcstate,exstate,calimatrix,which),max=len(v_rabi))
        vv, sv = await job.done()
        yield [j]*measure.n, vv, sv
        

async def phaseCali(measure,phase,mode='vbroadcast',dcstate=[],exstate=[],calimatrix=None):
    len_data, t = len(phase), np.array([phase]*measure.n).T
    qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]
    zname, z_awg = [f'ch{i}' for i in qubit_z.inst['z_ch']], measure.awg[qubit_z.inst['z_awg']]
    zch = qubit_z.inst['z_ch']

    dc = {i:0 for i in (dcstate+exstate)}
    dc[dcstate[0]] = 0.3
    current = await zManage(measure,dcstate=dc,qnum=len(dc),calimatrix=calimatrix)

    pulse = await cww.funcarg(cww.zWave_im,qubit_z,during=100e-9,shift=200e-9,delta_im=120e6,volt=0,imAmp=current[dcstate[0]])
    await cww.writeWave(measure,z_awg,zname,pulse)
    await cww.couldRun(measure,z_awg,zch,namelist=zname)

    task_ex = await executeEXwave(measure,cww.phaseCaliwave,exstate=exstate,output=True,t_run=300e-9)

    task_zex = await executeZseq(measure,cww.Z_sequence_im,len_data,True,mode=mode,dcstate=exstate,\
        v_or_t=phase,arg='phaseim',during=100e-9,shift=200e-9,volt=0,delta_im=120e6,imAmp=current[dcstate[0]])
    concurrence([{**task_ex[0],**task_zex[0]},{**task_ex[1],**task_zex[1]}])

    await measure.psg['psg_lo'].setValue('Output','ON')

    for i in range(300):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base  

async def freqCali(measure,t_run,mode='vbroadcast',dcstate=[],exstate=[]):
    len_data, t = len(t_run), np.array([t_run]*measure.n).T
    
    # task_ex = await executeEXwave(measure,cww.phaseCaliwave,exstate=exstate,output=True,t_run=150e-9)
    task_ex = await executeEXseq(measure,cww.Coherence_sequence,len_data,True,exstate,True,mode,\
        v_or_t=(t_run+100)/1e9,arg='t_run')

    task_zex = await executeZseq(measure,cww.Z_sequence,len_data,True,mode=mode,dcstate=exstate,\
        v_or_t=t_run/1e9,arg='during',shift=200e-9)
    concurrence([{**task_ex[0],**task_zex[0]},{**task_ex[1],**task_zex[1]}])

    await measure.psg['psg_lo'].setValue('Output','ON')

    for i in range(300):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base  

################################################################################
# Ramsey
################################################################################

async def Ramsey(measure,t_run,exstate=[]):

    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    # await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cww.couldRun(measure,awg_ex,chlist,namelist=namelist)

    for i in t_run:
        # pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=i/1e9)
        # await cww.writeWave(measure,awg_ex,name=namelist,pulse=pulselist)
        # await cww.couldRun(measure,awg_ex)
        task = await executeEXwave(measure,cww.coherenceWave,exstate=exstate,t_run=i/1e9)
        concurrence(task)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# SpinEcho
################################################################################

async def SpinEcho(measure,t_run,exstate=[]):
    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    # await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cww.couldRun(measure,awg_ex,chlist,namelist=namelist)
    for i in t_run:
        # pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=i/1e9)
        # await cww.writeWave(measure,awg_ex,name=namelist,pulse=pulselist)
        # await cww.couldRun(measure,awg_ex)
        task = await executeEXwave(measure,cww.coherenceWave,exstate=exstate,t_run=i/1e9)
        concurrence(task)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base

################################################################################
# spec crosstalk cali
################################################################################

async def single_cs(measure,t,len_data):
    for i in range(300):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s 

async def crosstalkSpec(measure,v_rabi,dcstate=[],exstate=[],comwave=False):
    qubit_zz = measure.qubits[dcstate[0]]
    zz_awg = measure.awg[qubit_zz.inst['z_awg']]
    zname, zch = [f'ch{i}' for i in qubit_zz.inst['z_ch']], qubit_zz.inst['z_ch']
    len_data, t = len(v_rabi), np.array([v_rabi]*measure.n).T
    
    qubit_ex = measure.qubits[exstate[0]]
    await measure.psg['psg_trans'].setValue('Frequency',qubit_ex.f_ex)
    await measure.psg['psg_trans'].setValue('Output','ON')
    task_z = await executeZseq(measure,cww.Z_sequence,len_data,comwave,dcstate=exstate,\
        v_or_t=v_rabi,arg='volt',during=(len(measure.t_new)/2.5/2e9+2000e-9),shift=200e-9)
    concurrence(task_z)

    await measure.psg['psg_lo'].setValue('Output','ON')
    for i in np.linspace(-1,1,11):

        pulse = await cww.funcarg(cww.zWave,qubit_zz,during=(len(measure.t_new)/2.5/2e9+2000e-9),volt=i,shift=200/1e9)
        await cww.writeWave(measure,zz_awg,zname,pulse,mark=False)
        await cww.couldRun(measure,zz_awg,namelist=zname,chlist=zch)
        job = Job(single_cs,(measure,t,len_data),avg=True,max=300,auto_save=False,no_bar=True)
        v_bias, s =await job.done()
        n = np.shape(s)[1]
        yield [i]*n, v_bias, s

################################################################################
# ramsey crosstalk cali
################################################################################

async def Rcscali_single(measure,v_rabi,t_col=0,exstate=[],dcstate=[],comwave=False):
    qubit_ex = measure.qubits[exstate[0]]
    ex_name = [f'ch{i}' for i in qubit_ex.inst['ex_ch']]
    ex_ch = qubit_ex.inst['ex_ch']
    ex_awg = measure.awg[qubit_ex.inst['ex_awg']]
    len_data, t = len(v_rabi), np.array([v_rabi]*measure.n).T
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')

    pulse1 = await cww.funcarg(cww.singleQgate,qubit_ex,axis='Xhalf',shift=(qubit_ex.pi_len+900e-9))
    pulse2 = await cww.funcarg(cww.singleQgate,qubit_ex,axis='AnyRhalf',phase=2*np.pi*0.25e6*900e-9)
    pulse = np.array(pulse1) + np.array(pulse2)
    await cww.writeWave(measure,ex_awg,ex_name,pulse)
    await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)

    task_z = await executeZseq(measure,cww.Z_sequence,len_data,comwave,dcstate=dcstate,\
        v_or_t=v_rabi,arg='volt',during=t_col*1e-9,shift=(qubit_ex.pi_len+150/1e9))
    concurrence(task_z)
    for i in range(300):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# 矫正比特间时序
################################################################################

async def qqTiming(measure,v_rabi,exstate=[],dcstate=[],comwave=False):
    qubit_ex = measure.qubits[exstate[0]]
    qubit_z = measure.qubits[dcstate[0]]
    printLog(qubit_z.asdict())
    printLog(qubit_ex.asdict(),mode='a')

    ex_name = [f'ch{i}' for i in qubit_ex.inst['ex_ch']]
    ex_ch = qubit_ex.inst['ex_ch']
    ex_awg = measure.awg[qubit_ex.inst['ex_awg']]
    len_data, t = len(v_rabi), np.array([v_rabi]*measure.n).T
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Frequency',(qubit_ex.f_ex+qubit_ex.delta_ex))

    pulse1 = await cww.funcarg(cww.singleQgate,qubit_ex,axis='Xhalf',shift=1000e-9)
    pulse2 = await cww.funcarg(cww.singleQgate,qubit_ex,axis='Xhalf',shift=500e-9)
    pulse = np.array(pulse1) + np.array(pulse2)
    await cww.writeWave(measure,ex_awg,ex_name,pulse)
    await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)

    task_z = await executeZseq(measure,cww.Z_sequence,len_data,comwave,dcstate=dcstate,\
        v_or_t=v_rabi/1e9,arg='shift',during=qubit_ex.pi_len,volt=0.3)
    concurrence(task_z)
    for i in range(300):
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base



################################################################################
# RTO_Notomo
################################################################################

async def RTO_Notomo(measure,t_run,exstate=[]):
    await cww.couldRun(measure,measure.awg['awgread'],[1,5])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for i in t_run:
        pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=500/1e9)
        await cww.writeWave(measure,awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(measure,awg_ex,chlist)
        ch_A, ch_B, I, Q = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base






async def singleZ(measure,t_T1,exstate=[],comwave=True):
    qubit = measure.qubits[exstate[0]]
    len_data, t = len(t_T1), np.array([t_T1]*measure.n).T
    if comwave:
        task = await executeEXseq(measure,cww.Rabi_sequence,len_data,comwave,exstate=exstate,v_or_t=t_T1/1e9,arg='shift')
        concurrence(task)
    await measure.psg['psg_lo'].setValue('Output','ON')
 
    pop_m = await seqReadpop(measure,len_data)
    yield 5000-qubit.pi_len*qubit.envelopename[1]/2e-9-t, np.array([pop_m]*measure.n).T

async def zPulse_pop(measure,t_T1,volt_init,height=0,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    
    printLog(qubit.asdict())

    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    pulse1 = await cww.funcarg(cww.zWave,qubit,during=2000/1e9,volt=height,shift=5000e-9,readvolt=0)
    pulse3 = await cww.funcarg(cww.zWave,qubit,during=0,volt=0,shift=0)
    await cww.couldRun(measure,z_awg,zch,zname)
    
    for i, j in enumerate(volt_init):
        
        pulse2 = await cww.funcarg(cww.zWave,qubit,during=7500/1e9,volt=j,shift=1500e-9,readvolt=0)
        # pulse = np.array(pulse1) + np.array(pulse2)
        pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse3)
        await cww.writeWave(measure,z_awg,zname,pulse)
        await cww.couldRun(measure,z_awg)
        comwave = True if i == 0 else False
        job = Job(singleZ,(measure,t_T1,exstate,comwave),auto_save=False,no_bar=True)
        t, pop = await job.done()

        yield [j], t, pop
   