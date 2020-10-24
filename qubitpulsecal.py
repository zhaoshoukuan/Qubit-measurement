import numpy as np
from scipy.optimize import leastsq
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate
from pyle.plotting import dstools
import scipy.optimize

def _getstepfunc(d, height, start=5, plot=False, ind=None, dep=None, w=0.01, timeconstants=2,returnData=False):
    if plot:
        dstools.plotDataset2Dexact(d, ind, dep, cmap=cm.get_cmap('rainbow'))
    d = d[np.argsort(d[:,0]),:]    
    boundaries = np.argwhere(np.diff(d[:,0]) > 0)[:,0]
    boundaries = np.hstack((0, boundaries, len(d)))
    n = len(boundaries)-1
    result = np.zeros((n, 2))
    result[:,0] = d[boundaries[:-1],0]
    for i in np.arange(n):
        p = d[boundaries[i]:boundaries[i+1],:]
        p = p[np.argsort(p[:,1]),:]
        result[i,1] = maximum(p[:,1], p[:,2], w=w)
    good = np.argwhere((abs(result[:,1]) < 0.2) * (result[:,0] > 0))[:,0]
    result = result[good,:]
    result = result[np.argwhere(result[:,0] >= start)[:,0],:] 
    if plot:
        plt.plot(result[:,0], result[:,1], 'w.')
    t = np.linspace(result[0,0], result[-1,0], 1000)
    def fitfunc(t, p):
        return (t > 0) * (p[0] + np.sum(p[1::2,None]*np.exp(-p[2::2,None]*t[None,:]), axis=0))
    def errfunc(p):
        return fitfunc(result[:,0], p) - result[:,1]
    p = np.zeros(2*timeconstants+1)
    p[2::2] = np.linspace(0.001, 0.3, timeconstants)
    p, _ = leastsq(errfunc, p,maxfev=70000)
    print 'Pulse relaxation:'
    for i in range(timeconstants):
        print '    amplitude: %g %%' % (100.0*p[1+2*i]/float(height))
        print '    time constant: %g ns' % (1.0/p[2+2*i])
    print ' RMS error: %g' % np.sqrt(np.average(errfunc(p)**2))
    t1 = t
    if t[0]>0:
        t1 = np.hstack([np.arange(0,t[0]-0.001,0.2),t])
    t2 = np.linspace(t1.min(), t1.max(), 3000)
    if plot:
        # plt.plot(t1, fitfunc(t1, p), 'k-')
        plt.plot(t2, fitfunc(t2, p), 'k-')
    p[0] = float(height)
    if returnData:
        return p, lambda t: fitfunc(t, p),result
    else:
        return p, lambda t: fitfunc(t, p)
        
def _getstepfunc_v2(d, height, start=0, fitLevel=10, tCut=200.0, plot=False, ind=None, dep=None, w=0.01, timeconstants=2,returnData=False):
    if plot:
        dstools.plotDataset2Dexact(d, ind, dep, cmap=cm.get_cmap('rainbow'))
    d = d[np.argsort(d[:,0]),:]    
    boundaries = np.argwhere(np.diff(d[:,0]) > 0)[:,0]
    boundaries = np.hstack((0, boundaries, len(d)))
    n = len(boundaries)-1
    result = np.zeros((n, 2))
    result[:,0] = d[boundaries[:-1],0]
    for i in np.arange(n):
        p = d[boundaries[i]:boundaries[i+1],:]
        p = p[np.argsort(p[:,1]),:]
        result[i,1] = maximum(p[:,1], p[:,2], w=w)
    good = np.argwhere((abs(result[:,1]) < 0.2) * (result[:,0] > 0))[:,0]
    index = np.argmin(np.abs(result[:,0]-tCut))
    result = result[good,:]
    result = result[np.argwhere(result[:,0] >= start)[:,0],:] 
    # print ('result', result[:,1])
    # print ('result', result[:,0])
    # result = scipy.interpolate.spline(result[:,0],result[:,1],result[:,0],order=20, kind='smoothest')[0:index+1]
    # print ('result', result)
    if plot:
        plt.plot(result[:,0], result[:,1], 'w.')
    
    para = np.polyfit(result[:,0], result[:,1], fitLevel)
    t = np.linspace(result[0,0], result[index-1,0], 1000)
    print ('tstop', result[index-1,0])
    def fitfunc(t, p):
        return (p[0] + np.sum(p[1::2,None]*np.exp(-p[2::2,None]*t), axis=0))*np.polyval(para,t)*(t <tCut)
    def errfunc(p):
        return fitfunc(result[:,0], p) - result[:,1]
    p = np.zeros(2*timeconstants+1)
    p[2::2] = np.linspace(0.001, 0.3, timeconstants)
    p, _ = leastsq(errfunc, p,maxfev=70000)
    tshift=5.0; sigma=0.8
    def smoothFuncATtCut(t,tCut,tshift):
        
        return (0.5-0.5*scipy.special.erf(sigma*(t-tCut+tshift)))*(0.5+0.5*scipy.special.erf(2.0*(t-result[0,0]+0.5)))
    
    
    
    print 'Pulse relaxation:'
    for i in range(timeconstants):
        print '    amplitude: %g %%' % (100.0*p[1+2*i]/float(height))
        print '    time constant: %g ns' % (1.0/p[2+2*i])
    print ' RMS error: %g' % np.sqrt(np.average(errfunc(p)**2))
    t1 = t
    # if t[0]>0:
        # t1 = np.hstack([np.arange(result[0,0],t[0]-0.001,0.2),t])
        # t1 = np.hstack([np.arange(0,t[0]-0.001,0.2),t])
    t2 = np.linspace(t1.min(), t1.max(), 3000)
    
    smoothData = fitfunc(t2,p)*smoothFuncATtCut(t2,tCut,tshift)
    timeFunData0 = smoothData
    if plot:
        # plt.plot(t1, fitfunc(t1, p), 'k-')
        plt.plot(t2, fitfunc(t2, p), 'k-')
        plt.plot(t2, timeFunData0, 'b--')
    p[0] = float(height)
    plt.figure()
    plt.plot(result[:,0], result[:,1], 'ko', mfc='none')
    plt.plot(t2, np.polyval(para,t2))
    if returnData:
        return p, lambda t: fitfunc(t, p),result
    else:
        return p, lambda t: fitfunc(t, p)       
        
        
        
        

        
        


def getstepfunc(ds, dataset=None, session=None, qubit=None, start=5, plot=False, w=0.01, timeconstants=2):
    d = dstools.getDataset(ds, dataset=dataset, session=session)
    ind, dep = ds.variables()
    height = ds.get_parameter('step height')
    p, func = _getstepfunc(d, height, start=start, plot=plot, ind=ind, dep=dep,
                           w=w, timeconstants=timeconstants)
    if qubit is not None:
        qubit['settling_rates'] = p[2::2]
        qubit['settling_amplitudes'] = p[1::2]/float(height)
    return p, func

def _getstepfuncWithPoly(d, height, start=5, plot=False, ind=None, dep=None, w=0.01, expConstantNum=2,returnData=False,polyNum=4):
    if plot:
        dstools.plotDataset2Dexact(d, ind, dep, cmap=cm.get_cmap('rainbow'))
    d = d[np.argsort(d[:,0]),:]    
    boundaries = np.argwhere(np.diff(d[:,0]) > 0)[:,0]
    boundaries = np.hstack((0, boundaries, len(d)))
    n = len(boundaries)-1
    result = np.zeros((n, 2))
    result[:,0] = d[boundaries[:-1],0]
    
    for i in np.arange(n):
        p = d[boundaries[i]:boundaries[i+1],:]
        p = p[np.argsort(p[:,1]),:]
        result[i,1] = maximum(p[:,1], p[:,2], w=w)
    good = np.argwhere((abs(result[:,1]) < 0.2) * (result[:,0] > 0))[:,0]
    result = result[good,:]
    result = result[np.argwhere(result[:,0] >= start)[:,0],:] 
    if np.abs(result[0,0]-result[1,0])<1e-10:
        result = result[1:,:]
    if plot:
        plt.plot(result[:,0], result[:,1], 'w.')
    
    def fitfunc(t, p):
        return (t > 0) * (p[0] + np.sum(p[1::2,None]*np.exp(-p[2::2,None]*t[None,:]), axis=0))
    def errfunc(p):
        return fitfunc(result[:,0], p) - result[:,1]
    p = np.zeros(2*expConstantNum+1)
    p[2::2] = np.linspace(0.001, 0.3, expConstantNum)
    p, _ = leastsq(errfunc, p, maxfev=70000)
    
    polyPara = np.polyfit(result[:50,0],errfunc(p)[:50],polyNum)
    print 'Pulse relaxation:'
    for i in range(expConstantNum):
        print '    amplitude: %g %%' % (100.0*p[1+2*i]/float(height))
        print '    time constant: %g ns' % (1.0/p[2+2*i])
    print ' RMS error: %g' % np.sqrt(np.average(errfunc(p)**2))
    t = np.arange(result[0,0], result[-1,0], 1.0)
    if t[0]>0:
        t1 = np.hstack([np.arange(0,t[0]-0.001,0.2),t])
    if plot:
        plt.plot(t1, fitfunc(t1, p), 'k-')
        plt.figure()
        plt.plot(result[:,0], fitfunc(result[:,0], p)-result[:,1], 'b.-')
        plt.plot(t1, np.polyval(polyPara,t1), 'r.-')
    p[0] = float(height)
    if returnData:
        return p,polyPara, lambda t: fitfunc(t, p),result
    else:
        return p,polyPara, lambda t: fitfunc(t, p)
        
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
    
                         
def fit_peak(x, y, is_plot=False, name='', w=0.01, peak_range0=0.3, is_boundary=True, close=None):

    pfunc = interpolate.UnivariateSpline(x, y, w=w)
    pfunc_der = pfunc.derivative()
    num = 2000
    x_sim = np.linspace(np.min(x), np.max(x), num)
    y_sim_der = pfunc_der(x_sim)
    
    bond_x = [x_sim[0], x_sim[-1]]
    peak_x = []
    dip_x = []
    for ii in range(0, num-1):
        if y_sim_der[ii] > 0.0 and y_sim_der[ii + 1] < 0.0:
            peak_x.append(x_sim[ii])
        if y_sim_der[ii] < 0.0 and y_sim_der[ii + 1] > 0.0:
            dip_x.append(x_sim[ii]) 
            
    bond_values = pfunc(bond_x)        
    peak_values = pfunc(peak_x)        
    dip_values = pfunc(dip_x)
    if close is None:
        if is_boundary:
            if len(peak_x) == 0:
                print 'No normal maximum value !'
                print 'There is only one dip in this function !'
                peak_value = np.max(bond_values)
                peak = bond_x[np.argmax(bond_values)]
            else:
                peak_max0 = np.max(peak_values)
                peak_max0_x = peak_x[np.argmax(peak_values)]
                
                peak_max1 = np.max(bond_values)
                peak_max1_x = bond_x[np.argmax(bond_values)]
                
                peak_value = max(peak_max0, peak_max1)
                peak = peak_max0_x if peak_max0 > peak_max1 else peak_max1_x  
        else:
            if len(peak_x) == 0:
                peak = np.mean(x_sim)
                peak_value = pfunc(peak)
            else:
                peak_value = np.max(peak_values)
                peak = peak_x[np.argmax(peak_values)]
    else:
        if is_boundary:
            if len(peak_x) == 0:
                print 'No normal maximum value !'
                print 'There is only one dip in this function !'
                peak_value = np.max(bond_values)
                peak = bond_x[np.argmax(bond_values)]
            else:
                peak_max0 = np.max(peak_values)
                peak_max0_x = peak_x[np.argmax(peak_values)]
                
                peak_max1 = np.max(bond_values)
                peak_max1_x = bond_x[np.argmax(bond_values)]
                
                peak_value = max(peak_max0, peak_max1)
                peak = peak_max0_x if peak_max0 > peak_max1 else peak_max1_x  
        else:
            if len(peak_x) == 0:
                peak = np.mean(x_sim)
                peak_value = pfunc(peak)
            else:
                peak_value = np.max(peak_values)
                peak = peak_x[np.argmax(peak_values)]
    
    
    extremes = np.sort(np.append(bond_x, dip_x)) 
    diffs = np.sort(np.abs(peak - extremes))
    peak_range = peak_range0 if np.max(diffs[0:2]) < peak_range0 else 0.8 * np.max(diffs[0:2])
    # print 'diffs:',np.max(diffs[0:2])    
    # print 'range0:',peak_range0
    if is_plot:
        plt.figure(figsize=[8,4])
        plt.grid('on');
        plt.plot(x, y, '.r')
        plt.plot(x, pfunc(x), 'r')
        plt.plot(bond_x, bond_values, 'sg')
        plt.plot(peak_x, peak_values, 'sk')
        plt.plot(dip_x, dip_values, 'sb')
        plt.xlabel(name, size=18)
        plt.ylabel('Prob.', size=18)
        plt.subplots_adjust(bottom=0.15, left=0.12)
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.title(name + ': ' + str(np.round(peak, 2)), size=18)
        plt.xlim(np.min(x), np.max(x))
        plt.show()  
    return peak, peak_range, peak_value  