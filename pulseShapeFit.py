# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:57:08 2020

@author: xukai
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import scipy.signal
from pylab import *
sys.path.insert(0,'N:\\DataTakingEclipseV1\\pyle\\')
sys.path.insert(0,'N:\\DataTakingEclipseV1\\pyle\\pyle')
sys.path.insert(4,'N:\\DataTakingEclipseV1\\datataking\\')
from pyle import envelopes as env
import scipy.io
import scipy.interpolate
import scipy.optimize
from pyle.plotting import dstools as ds
import conf
from pyle.dataking import dataProcess as dp
try:
    cxn = conf.cxn
    conf.switchSession(user='lihao',session='20200916_20qubit_sample_singlequbit')
    s = conf.s
except:
    cxn = labrad.connect()
    conf.switchSession(user='lihao')
    s = conf.s
dv = cxn.data_vault
saveDir = 'D:\\experiment\\projects\\spinSqueezing\\'
def maximum(x, y, w=0.01):
    n = len(x)
    dx = (x[-1] - x[0]) / (n-1)
    f = np.exp(-np.linspace(-n*dx/w, n*dx/w, 2*n-1)**2)
    smooth = np.real(np.convolve(y, f, 'valid')) / np.real(np.convolve(1-np.isnan(y), f, 'valid'))
    #plt.plot(smooth)
    i = np.argmax(smooth)
    if i > 0 and i < n-1:
        yl, yc, yr = smooth[i-1:i+2]
        d = 2.0 * yc - yl - yr
        if d <= 0:
            return 0
        d = 0.5 * (yr - yl) / d
        d = np.clip(d, -1, 1)
    else:
        d = 0
        print 'warning: no maximum found'
    return (1-abs(d))*x[i] + d*(d>0)*x[i+1] - d*(d<0)*x[i-1]
#%%
#process pulseshapData 1st
dataset = 82#66#q18:329; q10:332; q13:338; q6:341; q9:370; q14:54; q17:56
#session = ['', 'xukai', '20200804_20qubit_sample']
session = ['', 'lihao', '20200830_20qubit_sample_pulseShape']
qubit = 'q7'
updatePara = False
polyNum = 24
timeconstants = 2
tCut = 225.0
tshift = 5

polyRatio = 1.0
sigma = 0.4
tStart = 3
tcuts = [0,10e3]
tcuts1 = None
expNum = 1

assert(expNum==1 and polyNum<=25 and timeconstants<=3)
pulsedata = ds.getDataset(dv,dataset,session)
pulsePara = ds.getParameters(dv,dataset,session)
measurei = int(np.squeeze(pulsePara['measure']))
qubit1 = pulsePara['config'][measurei]
if qubit is None:
    qubit = qubit1
if tcuts is not None:
    pulsedata = pulsedata[pulsedata[:,0]>=tcuts[0],:]
    pulsedata = pulsedata[pulsedata[:,0]<tcuts[1],:]
if tcuts1 is not None:
    pulsedata = np.vstack([pulsedata[pulsedata[:,0]<=tcuts1[0],:],pulsedata[pulsedata[:,0]>=tcuts1[1],:]])
height = pulsePara['step height']

ind=['1','2']; dep=['3']
ds.plotDataset2Dexact(pulsedata, ind, dep, cmap=plt.cm.get_cmap('rainbow'))
pulsedata = pulsedata[np.argsort(pulsedata[:,0]),:]    
boundaries = np.argwhere(np.diff(pulsedata[:,0]) > 0)[:,0]
boundaries = np.hstack((0, boundaries, len(pulsedata)))
n = len(boundaries)-1
result = np.zeros((n, 2))
result[:,0] = pulsedata[boundaries[:-1],0]
for i in np.arange(n):
    p = pulsedata[boundaries[i]:boundaries[i+1],:]
    p = p[np.argsort(p[:,1]),:]
    result[i,1] = maximum(p[:,1], p[:,2])
#    result[i,1] = dp.fit_peak(p[:,1]+1e-10*(np.random.rand(len(p[:,1]))-0.5), p[:,2], s=0.01)[0]
good = np.argwhere((abs(result[:,1]) < 0.2) * (result[:,0] > 0))[:,0]
result = result[good,:]
result = result[np.argwhere(result[:,0] >= tStart)[:,0],:] 
if np.abs(result[0,0]-result[1,0])<1e-10:
    result = result[1:,:]
data = result    
#data[data[:,0]>173.0,:]+=0.00789
if np.abs(data[0,0]-data[1,0])<1e-10:
    data = data[1:,:]
data_smooth = scipy.interpolate.spline(data[:,0],data[:,1],data[:,0],order=10)

def fitfunc(t, p):
    return (t > 0) * (p[0] + np.sum(p[1::2,None]*np.exp(-p[2::2,None]*t[None,:]), axis=0))
def errfunc(p):
    return (fitfunc(data[:,0], p) - data[:,1])*(1.0+0.5*scipy.special.erf(0.4*(data[:,0]-tCut)))**5
p0 = np.zeros(2*timeconstants+1)
p0[2::2] = np.linspace(0.001, 0.3, timeconstants)
p, _ = scipy.optimize.leastsq(errfunc, p0)
ts = np.arange(0,data[-1,0]*2,0.02)
plt.plot(data[:,0],data[:,1],'bo')
plt.figure()
subplot(211)
plt.plot(data[:,0],data[:,1],'bo')
plt.plot(ts,fitfunc(ts,p),'k-')

data1 = np.copy(data)
restData = data[:,1]-fitfunc(data[:,0], p)
restData = restData[data1[:,0]<=tCut]
data1 = data1[data1[:,0]<=tCut,:]
def fitfunc1(t, p):
    pExp = p[:expNum*2]
    pPoly = p[expNum*2:]
    if np.iterable(t):
        return np.sum(pExp[0::2,None]*np.exp(-pExp[1::2,None]*t[None,:]), axis=0)*np.polyval(pPoly,t)*(t<=tCut+20)
    else:
        return np.sum(pExp[0::2]*np.exp(-pExp[1::2]*t))*np.polyval(pPoly,t)*(t<=tCut)
def errfunc1(p):
    return fitfunc1(data1[:,0], p) - restData
def smoothFuncATtCut(ts,tCut,tshift):    
    return (0.5-0.5*scipy.special.erf(sigma*(ts-tCut+tshift)))*(0.5+0.5*scipy.special.erf(2.0*(ts-data[0,0]+0.5)))
pExp0 = np.zeros(2*expNum)
pExp0[1::2] = np.linspace(0.001, 0.3, expNum)
pPoly0 = np.zeros(polyNum)
pPoly0[-1] = 1.0
pAll0 = np.hstack([pExp0,pPoly0])
p2, _ = scipy.optimize.leastsq(errfunc1, pAll0)
smoothData = fitfunc1(ts,p2)*smoothFuncATtCut(ts,tCut,tshift)*polyRatio
timeFunData0 = smoothData+fitfunc(ts,p)
paras= {'p':p,'p2':p2,'tCut':tCut,'tshift':tshift,'sigma':sigma}
subplot(212)
plt.plot(data1[:,0],restData,'bo')
plt.plot(ts,smoothData,'r-')
plt.xlabel('Time (ns)')
plt.grid(True)
subplot(211)
plt.plot(ts,timeFunData0,'r-')
plt.grid(True)
p2 = np.asarray(p2)
p = np.asarray(p)
p_uniform = np.copy(p)
p2_uniform = np.copy(p2)
p_uniform[1::2] = p_uniform[1::2]/height
p2_uniform[0] = p2[0]/height*polyRatio
if updatePara==True:  
    len0 = np.int(np.max([len(qubit),len(qubit1)]))
    assert(qubit1[:len0]==qubit[:len0])
    s[qubit]['expAmpRates'] = p_uniform[1:]
    s[qubit]['polyParas'] = p2_uniform
    s[qubit]['delayParas'] = np.array([tCut,tshift,sigma])
 
# #%%
# #process pulseshapData 2nd
# dataset = 170#234#q18:329; q10:332; q13:338; q6:341; q9:370; q14:54; q17:56
# #session = ['', 'xukai', '20200804_20qubit_sample']
# session = ['', 'lihao', '20200906_20qubit_sample']
# qubit = 'q20'
# updatePara = True
# polyNum = 20
# tCut = 160.0
# deconvoloveData = False
# maxRelaxTime = 8.0

# tshift = 5
# polyRatio = 1.0
# sigma = 0.4
# tStart = 2
# tcuts = [0,10e3]
# tcuts1 = None
# expNum = 1
# timeconstants = 2
# assert(expNum==1 and polyNum<=25 and timeconstants==2 and maxRelaxTime<800.01) 
# pulsedata,xs,ys = dp.getMatrixForMultiLens(dv,dataset,session,endData=False)
# pulsePara = ds.getParameters(dv,dataset,session)
# measurei = int(np.squeeze(pulsePara['measure']))
# qubit1 = pulsePara['config'][measurei]
# if qubit is None:
#     qubit = qubit1
# height = pulsePara['step height']
# if maxRelaxTime>np.max(xs[-1])/2.0:
#     print('max relax time should be smaller than 1/2 of the total mesure time of the pulsehshape, which is %s ns'%(np.max(np.max(xs[-1])/2.0)))
#     assert(maxRelaxTime<np.max(np.max(xs[-1])/2.0))
# ind=['1','2']; dep=['3']
# ds.plotDataset2Dexact(ds.getDataset(dv,dataset,session), ind, dep, cmap=plt.cm.get_cmap('rainbow'))
# n = len(xs)
# result = []
# for i in np.arange(n):
#     if xs[i][0]<tcuts[0] or xs[i][0]>tcuts[1]:
#         continue
#     assert(np.sum(np.abs(np.diff(xs[i])))<1e-10)
#     expDatai = pulsedata[i]
#     ysi = ys[i]
#     yindexs = np.argsort(ysi)
#     expDatai = expDatai[yindexs,:]
#     ysi = ysi[yindexs]
# #    result[i,1] = maximum(p[:,1], p[:,2], w=0.01)
#     is_plot=True if i%10==10000 else False
#     fitx = dp.fit_peak(ysi, expDatai[:,0], s=0.07, is_plot=is_plot)[0]
#     if expDatai.shape[1]==2:
#         base0 = dp.fit_peak(ysi,expDatai[:,1], s=0.07, is_plot=is_plot)[0]
#         fitx -= base0
#     result.append([xs[i][0],fitx])
# result = np.asarray(result)
# good = np.argwhere((abs(result[:,1]) < 0.2) * (result[:,0] > 0))[:,0]
# result = result[good,:]
# result = result[np.argwhere(result[:,0] >= tStart)[:,0],:] 
# if np.abs(result[0,0]-result[1,0])<1e-10:
#     result = result[1:,:]
# tmax = np.max(result[:,0])
# data = result    
# ts = np.arange(0,data[-1,0],0.2)
# if deconvoloveData:
#     data = deconvolvePulseShape(ts,data,tcut=500,w=10.0)
# if np.abs(data[0,0]-data[1,0])<1e-10:
#     data = data[1:,:]

# def fitfunc(t, p):
#     return (t >= 0) * (p[0] + np.sum(p[1::2,None]*np.exp(-p[2::2,None]*t[None,:]), axis=0))
# def errfunc(p):
#     return (fitfunc(data[:,0], p) - data[:,1])*(1.0+0.5*scipy.special.erf(0.4*(data[:,0]-tCut)))**5
# errfunc2 = lambda p:np.sum(np.abs(errfunc(p)**2))
# p0 = np.zeros(2*timeconstants+1)
# p0[2::2] = np.linspace(0.001, 0.3, timeconstants)
# #p, _ = scipy.optimize.leastsq(errfunc, p0)
# bnds = [(-0.5,0.5),(-0.5,0.5),(1.0/maxRelaxTime,np.inf),(-0.5,0.5),(maxRelaxTime,np.inf)]
# p = scipy.optimize.fmin_l_bfgs_b(errfunc2,p0,bounds=bnds,approx_grad=True)[0]
# for idxi,pii in enumerate(p[2::2]):
#     print(pii)
#     if 1.0/pii>800.0:
#         p[2::2][idxi] = 1/800.0

# plt.plot(data[:,0],data[:,1],'bo')
# plt.figure()
# plt.subplot(211)
# plt.plot(data[:,0],data[:,1],'bo')
# plt.plot(ts,fitfunc(ts,p),'k-')

# data1 = np.copy(data)
# restData = data[:,1]-fitfunc(data[:,0], p)
# restData = restData[data1[:,0]<=tCut]
# data1 = data1[data1[:,0]<=tCut,:]
# def fitfunc1(t, p):
#     pExp = p[:expNum*2]
#     pPoly = p[expNum*2:]
#     if np.iterable(t):
#         return np.sum(pExp[0::2,None]*np.exp(-pExp[1::2,None]*t[None,:]), axis=0)*np.polyval(pPoly,t)*(t<=tCut+20)
#     else:
#         return np.sum(pExp[0::2]*np.exp(-pExp[1::2]*t))*np.polyval(pPoly,t)*(t<=tCut)
# def errfunc1(p):
#     return fitfunc1(data1[:,0], p) - restData
# def smoothFuncATtCut(ts,tCut,tshift):    
#     return (0.5-0.5*scipy.special.erf(sigma*(ts-tCut+tshift)))*(0.5+0.5*scipy.special.erf(2.0*(ts-data[0,0]+0.5)))
# pExp0 = np.zeros(2*expNum)
# pExp0[1::2] = np.linspace(0.001, 0.3, expNum)
# pPoly0 = np.zeros(polyNum)
# pPoly0[-1] = 1.0
# pAll0 = np.hstack([pExp0,pPoly0])
# p2, _ = scipy.optimize.leastsq(errfunc1, pAll0)
# smoothData = fitfunc1(ts,p2)*smoothFuncATtCut(ts,tCut,tshift)*polyRatio
# timeFunData0 = smoothData+fitfunc(ts,p)
# paras= {'p':p,'p2':p2,'tCut':tCut,'tshift':tshift,'sigma':sigma}
# subplot(212)
# plt.plot(data1[:,0],restData,'bo')
# plt.plot(ts,smoothData,'r-')
# plt.xlabel('Time (ns)')
# plt.grid(True)
# subplot(211)
# plt.plot(ts,timeFunData0,'r-')
# plt.grid(True)
# p2 = np.asarray(p2)
# p = np.asarray(p)
# p_uniform = np.copy(p)
# p2_uniform = np.copy(p2)
# p_uniform[1::2] = p_uniform[1::2]/height
# p2_uniform[0] = p2[0]/height*polyRatio
# if updatePara==True:   
#     len0 = np.int(np.max([len(qubit),len(qubit1)]))
#     assert(qubit1[:len0]==qubit[:len0])
#     sd = s[qubit]['delayParas']
    
#     para_2nd = np.hstack([p_uniform[1:],p2_uniform,np.array([tCut,tshift,sigma])])
#     s[qubit]['delayParas'] = np.hstack([sd[:3],para_2nd])
#     print ('updated!!!')
# #    expAmpRates = d.get('expAmpRates',[])+[d.get('acSettlingTime',1e10*ns)['ns']]
# #    polyParas = d.get('polyParas',[1.0,0.0,1.0,0.0,0.0])
# #    delayParas = d.get('delayParas',[0.0,0.0,1.0])#[tCut,tShif,sigma]
    
#%%
qubit = 'q17'
samplingRate = 1.0
nrfft = 2049
expAmpRates = s[qubit]['expAmpRates'].asarray
polyParas = s[qubit]['polyParas'].asarray
delayParasAND2nd = s[qubit]['delayParas'].asarray
nfft = 2*(nrfft-1)
freqs = np.linspace(0, nrfft*1.0/nfft*samplingRate,nrfft, endpoint=False)
i_two_pi_freqs = 2j*np.pi*freqs
pExp0 = expAmpRates
pExp1 = polyParas[:2]
pPoly = polyParas[2:]
tCut,tShift,sigma1 = delayParasAND2nd[:3]
tlist = np.arange(2*(nrfft-1),dtype=float)/samplingRate
timeFunData = np.sum(pExp0[0::2,None]*np.exp(-pExp0[1::2,None]*tlist[None,:]), axis=0)
timeFunData += pExp1[0]*np.exp(-pExp1[1]*tlist)*np.polyval(pPoly,tlist)*(tlist<=tCut+20)*\
    (0.5-0.5*scipy.special.erf(sigma1*(tlist-tCut+tShift)))*(0.5+0.5*scipy.special.erf(4.0*(tlist-3+0.5)))
timeFunDataf = np.fft.rfft(timeFunData)
# scipy.io.savemat('N:\\serversForMeaure\\ghzdac\\timeFunData.mat',{'data':timeFunData})
precalc = 1.0/(1.0+timeFunDataf*i_two_pi_freqs/samplingRate)
if len(delayParasAND2nd)>3:
    para2nd = delayParasAND2nd[3:]
    pExp0_2nd = para2nd[:4]
    pExp1_2nd = para2nd[4:6]
    pPoly_2nd = para2nd[6:-3]
    tCut2,tShift2,sigma2 = para2nd[-3:]
    timeFunData2 = np.sum(pExp0_2nd[0::2,None]*np.exp(-pExp0_2nd[1::2,None]*tlist[None,:]), axis=0)
    timeFunData2 += pExp1_2nd[0]*np.exp(-pExp1_2nd[1]*tlist)*np.polyval(pPoly_2nd,tlist)*(tlist<=tCut2+20)*(0.5-0.5*scipy.special.erf(sigma2*(tlist-tCut2+tShift2)))*(0.5+0.5*scipy.special.erf(4.0*(tlist-3+0.5)))
    timeFunDataf2 = np.fft.rfft(timeFunData2)
    precalc = precalc/(1.0+timeFunDataf2*i_two_pi_freqs/samplingRate)

plt.figure(10)
plt.plot(tlist,timeFunData+timeFunData2,'.-',label=qubit)
plt.legend()
plt.grid()



#%%
dataBefore = np.fft.irfft(np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/signalBeforeCo.mat')['data']))
dataAfter = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/signalAfterCo.mat')['data'])
plt.figure()
plt.plot(dataBefore,'.-')
plt.plot(dataAfter,'.-')
plt.grid(True)

#%%
dataAfter1z = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11Zdata_0.25us.mat')['signal'])
dataAfter2z = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11Zdata_0.26us.mat')['signal'])
dataAfter1xy = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11xydata_0.25us.mat')['signal'])
dataAfter2xy = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11xydata_0.26us.mat')['signal'])
plt.figure()
plt.plot(dataAfter1xy[:,1],'.-',label='XY,0.25 us')
plt.plot(dataAfter2xy[:,1],'.-',label='XY,0.26 us')
plt.plot(dataAfter1z,'.-',label='Z,0.25 us')
plt.plot(dataAfter2z,'.-',label='Z,0.26 us')
plt.legend()
plt.xlabel('Time')
#%%
dataAfter1z = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11Zdata_0.24us_0delay.mat')['signal'])
dataAfter2z = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11Zdata_0.24us_20delayv2.mat')['signal'])
dataAfter1xy = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11xydata_0.24us_0delay.mat')['signal'])
dataAfter2xy = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11xydata_0.24us_20delayv2.mat')['signal'])
plt.figure()
plt.plot(dataAfter1xy[:,1],'.-',label='XY,0.24 us,0 delay')
plt.plot(dataAfter2xy[:,1],'.-',label='XY,0.24 us,20 delay')
plt.plot(dataAfter1z,'.-',label='Z,0.24 us,0 delay')
plt.plot(dataAfter2z,'.-',label='Z,0.24 us,20 delay')
plt.legend()
plt.xlabel('Time')

#%%
samplingRate = 2.0
#dataAfter1z = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11ZdataOs0.0.mat')['signal'])
#dataAfter1z1 = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q17ZdataOs0.0.mat')['signal'])
dataAfter1z = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11Zdata.mat')['signal'])
dataAfter1z1 = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q17Zdata.mat')['signal'])

#dataAfter1z6 = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q11Zdata.mat')['signal'])
#dataAfter1z16 = np.squeeze(scipy.io.loadmat('N:/serversForMeaure/ghzdac/q17Zdata.mat')['signal'])
plt.figure(10)
plt.plot(np.arange(len(dataAfter1z))/float(samplingRate),dataAfter1z,'b.--',label='q11 z')
plt.plot(np.arange(len(dataAfter1z))/float(samplingRate),dataAfter1z1,'r.--',label='q17 z')
#plt.plot(dataAfter1z6,'b^-',label='q11 z +6')
#plt.plot(dataAfter1z16,'r^-',label='q17 z +6')
plt.legend()
plt.xlabel('Time (ns)')
plt.grid(True)
#%%
tcut = 2000
ts1 = ts[:tcut]
timeFunData1 = np.copy(timeFunData0[:tcut])
w = 7.0
sigma = w / np.sqrt(8*np.log(2))
impulse_response = 1.0 * np.exp(-(ts1-1*w)**2/(2*sigma**2))
impulse_response = impulse_response[ts1<2*w]
impulse_response = impulse_response/np.trapz(impulse_response,dx=1)
timeFunData1 -= timeFunData1[-1]
#timeFunData1[0] = 0.0
singleAfterConvolve = convolveOwn(impulse_response,timeFunData1)
singleAfterConvolveChop = singleAfterConvolve[:len(ts1)]
#singleAfterConvolveBack1 = deconvolveOwn(np.hstack([timeFunData1,[timeFunData1[-1]]*(len(impulse_response)-1)]), impulse_response)
#singleAfterConvolveBack, remainder = scipy.signal.deconvolve(singleAfterConvolve+np.random.normal(0.0,0.000,len(singleAfterConvolve)), impulse_response)
singleAfterConvolveBack = deconvolveOwn(singleAfterConvolve, impulse_response)

indexStart = np.argmin(np.abs(ts1-w))
singleAfterConvolveBack1 = timeFunData1+(timeFunData1-singleAfterConvolve[indexStart:][:len(timeFunData1)])
singleAfterConvolve2 = convolveOwn(impulse_response,singleAfterConvolveBack1)
singleAfterConvolve2 = singleAfterConvolve2[:len(timeFunData1)]
plt.figure(1)
plt.subplot(211)
plt.plot(ts1[ts1<2*w],impulse_response,'.-')
plt.grid(True)
plt.subplot(212)
plt.plot(ts1,timeFunData1,'o',label='original')
plt.plot(ts1-w,singleAfterConvolveChop,'.-',label='convolve')
#plt.plot(ts1,singleAfterConvolveBack,'.-',label='convolve + deconvolve')
plt.plot(ts1,singleAfterConvolveBack1,'.-',label='deconvolve')
plt.plot(ts1-w,singleAfterConvolve2,'.-',label='deconvolve+convolve')
plt.legend()
plt.grid(True)

#%%
tsExp,timeFunDataExp = result[:,0],result[:,1]
tcut = 5000
ts1 = ts[:tcut]
timeFunData1 = np.interp(ts1,tsExp,timeFunDataExp)
w = 10.0
sigma = w / np.sqrt(8*np.log(2))
impulse_response = 1.0 * np.exp(-(ts1-1*w)**2/(2*sigma**2))
impulse_response = impulse_response[ts1<2*w]
impulse_response = impulse_response/np.trapz(impulse_response,dx=1)
timeFunData1 -= timeFunData1[-1]
#timeFunData1[0] = 0.0
singleAfterConvolve = convolveOwn(impulse_response,timeFunData1)
singleAfterConvolveChop = singleAfterConvolve[:len(ts1)]
#singleAfterConvolveBack1 = deconvolveOwn(np.hstack([timeFunData1,[timeFunData1[-1]]*(len(impulse_response)-1)]), impulse_response)
#singleAfterConvolveBack, remainder = scipy.signal.deconvolve(singleAfterConvolve+np.random.normal(0.0,0.000,len(singleAfterConvolve)), impulse_response)
singleAfterConvolveBack = deconvolveOwn(singleAfterConvolve, impulse_response)

indexStart = np.argmin(np.abs(ts1-w))
singleAfterConvolveBack1 = timeFunData1+(timeFunData1-singleAfterConvolve[indexStart:][:len(timeFunData1)])
singleAfterConvolve2 = convolveOwn(impulse_response,singleAfterConvolveBack1)
singleAfterConvolve2 = singleAfterConvolve2[:len(timeFunData1)]

plt.figure(1)
plt.subplot(211)
plt.plot(ts1[ts1<2*w],impulse_response,'.-')
plt.grid(True)
plt.subplot(212)
plt.plot(ts1,timeFunData1,'o',label='original')
plt.plot(ts1-w,singleAfterConvolveChop,'.-',label='convolve')
#plt.plot(ts1,singleAfterConvolveBack,'.-',label='convolve + deconvolve')
plt.plot(ts1,singleAfterConvolveBack1,'.-',label='deconvolve')
plt.plot(ts1-w,singleAfterConvolve2,'.-',label='deconvolve+convolve')
plt.legend()
plt.grid(True)
#%%
def convolveOwn(impulse_response,data):
    convData = []
    lenr = len(impulse_response)
    lenData = len(data)
    convLen = lenr+lenData-1
#    data1 = np.hstack([data[-lenr+1:],data,data[:lenr+1]])    
    data1 = np.hstack([[0.0]*(lenr-1),data,[0.0]*lenr])    
    convData = []
    sumr = np.trapz(impulse_response,dx=1)
    for idx in range(convLen):
        convi = np.sum(impulse_response*data1[idx:idx+lenr])
        if idx<lenr:
            convi = convi/np.sum(impulse_response[-1::-1][:idx+1])*sumr
        convData.append(convi)
    convData = np.asarray(convData)
    return convData
    
def deconvolveOwn(data,impulse_response,checkNum=10):
    data1 = np.copy(data)
    dataLen = len(data1)
    lenr = len(impulse_response)
    impulse_responseInv = impulse_response[-1::-1]
    sumr = np.trapz(impulse_response,dx=1)
    deconvLen = dataLen-lenr+1
    deconvData = []
    for idx in range(lenr):
        data1[idx] = data1[idx]*np.sum(impulse_responseInv[:idx+1])/sumr
    deconvData.append(data1[0]/impulse_response[-1])
    for idx in range(1,lenr):
        deLen = len(deconvData)
        vi = data1[idx]-np.sum(np.asarray(deconvData)*impulse_response[-1-deLen:-1])
        deconvi = vi/impulse_response[-1]
        if len(deconvData)>10:
            diffi = np.diff(deconvData[-checkNum:])
            if np.sum(np.abs(diffi))<1e-7:
                deconvi = 0.0
        deconvData.append(deconvi)
    for idx in range(lenr,deconvLen):
        vi = data1[idx]-np.sum(impulse_response[:-1]*deconvData[-(lenr-1):])
        deconvi = vi/impulse_response[-1]
        if len(deconvData)>10:
            diffi = np.diff(deconvData[-checkNum:])
            if np.sum(np.abs(diffi))<1e-7:
                deconvi = 0.0
        deconvData.append(deconvi)
    deconvData = np.asarray(deconvData)
    return deconvData
    
def deconvolvePulseShape(ts,result,tcut=500,w=7.0):
    tsExp,timeFunDataExp = result[:,0],result[:,1]   
    ts1 = ts[ts<tcut]
    timeFunData1 = np.interp(ts1,tsExp,timeFunDataExp)
    
    sigma = w / np.sqrt(8*np.log(2))
    impulse_response = 1.0 * np.exp(-(ts1-1*w)**2/(2*sigma**2))
    impulse_response = impulse_response[ts1<2*w]
    impulse_response = impulse_response/np.trapz(impulse_response,dx=1)
    tOff = timeFunData1[-1]
    timeFunData1 -= tOff

    singleAfterConvolve = convolveOwn(impulse_response,timeFunData1)
    indexStart = np.argmin(np.abs(ts1-w))
    singleAfterConvolveBack1 = timeFunData1+(timeFunData1-singleAfterConvolve[indexStart:][:len(timeFunData1)])
    singleAfterConvolve2 = convolveOwn(impulse_response,singleAfterConvolveBack1)
    singleAfterConvolve2 = singleAfterConvolve2[:len(timeFunData1)]
    singleAfterConvolve2 = singleAfterConvolve2+tOff
    timeFunDataExpDe = np.interp(tsExp,ts1,singleAfterConvolveBack1)
    result1 = np.array([tsExp,timeFunDataExpDe]).T
    return result1

fft