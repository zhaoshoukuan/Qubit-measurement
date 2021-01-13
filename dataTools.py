import numpy as np
import scipy as sp 
import sympy as sy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


xvar = sy.Symbol('x',real=True)


def nearest(y,target,x=None):
    if x is None:
        index = np.argmin(np.abs(y-target))
        return index
    else:
        index = np.argmin(np.abs(y-target))
        return index, x[index]

# def specfunc2data(func,)
def specshift(funclist,fc,bias=0,side='lower'):
    func,voffset, vperiod, ejs, ec, d = funclist
    v = np.linspace(-vperiod/2,vperiod/2,50001) + voffset
    y = sy.lambdify(xvar,func,'numpy')
    if side == 'lower':
        f01 = y(v)[v<voffset]
        index, vtarget = nearest(f01,fc,v[v<voffset])
        # vnew = v - vtarget
        # _, fnew = nearest(vnew,bias,y(v))
        return y(vtarget+bias), (vtarget+bias)
    if side == 'higher':
        f01 = y(v)[v>voffset]
        index, vtarget = nearest(f01,fc,v[v>voffset])
        return y(vtarget+bias), (vtarget+bias)

def biasshift(funclist,fc,fshift=0,side='lower'):
    func,voffset, vperiod, ejs, ec, d = funclist
    v = np.linspace(-vperiod/2,vperiod/2,50001) + voffset
    y = sy.lambdify(xvar,func,'numpy')
    if np.max(fc+fshift)>np.max(y(v)):
        raise('too big')
    if side == 'lower':
        vnew = v[v<voffset]
        f01 = y(v)[v<voffset]
        finterp = sp.interpolate.interp1d(f01,vnew)
        index, vtarget = nearest(f01,fc,vnew)
        return finterp(fc+fshift)-vtarget
    if side == 'higher':
        vnew = v[v>voffset]
        f01 = y(v)[v>voffset]
        finterp = sp.interpolate.interp1d(f01,vnew)
        index, vtarget = nearest(f01,fc,vnew)
        return finterp(fc+fshift)-vtarget

def dresscali(funclist,dressenergy,fc,imAmp=0,side='lower'):

    fnew = dressenergy(imAmp)-f_ex/1e9
    bias_offset = dt.biasshift(specfuncz,f_ex/1e9,fnew,'lower')

def vTophi(funclist,T_bias,fc,side='lower'):
    func,voffset, vperiod, ejs, ec, d = funclist
    vperiod, voffset = T_bias
    tmp_s = np.pi*(xvar-voffset)/vperiod
    func = sy.sqrt(8*ejs*ec*sy.Abs(sy.cos(tmp_s))*sy.sqrt(1+d**2*sy.tan(tmp_s)**2))-ec
    v = np.linspace(-vperiod/2,vperiod/2,50001) + voffset
    y = sy.lambdify(xvar,func,'numpy')
    if side == 'lower':
        vnew = v[v<voffset]
        f01 = y(v)[v<voffset]
        # finterp = sp.interpolate.interp1d(f01,vnew)
        index, vtarget = nearest(f01,fc,vnew)
        return vtarget
    if side == 'higher':
        vnew = v[v>voffset]
        f01 = y(v)[v>voffset]
        # finterp = sp.interpolate.interp1d(f01,vnew)
        index, vtarget = nearest(f01,fc,vnew)
        return vtarget

def classify(measure,s_st,target=None,predictexe=True):
    
    num = measure.n//2+measure.n%2
    name = ''
    for i in measure.qubitToread:
        name += i
    if target is not None:
        name = f'q{target+1}'
    fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,4*num))
    n = measure.n if target is None else 1
    if predictexe:
        for i in range(n):
            i = target if target is not None else i
            s_off, s_on = s_st[0,:,i], s_st[1,:,i]
            S = list(s_off) + list(s_on)
            x,z = np.real(S), np.imag(S)
            d = list(zip(x,z))
            kmeans = KMeans(n_clusters=2,max_iter=100,tol=0.001)
            kmeans.fit(d)
            measure.predict[measure.qubitToread[i]] = kmeans.predict
            y = kmeans.predict(d)
            print(list(y).count(1)/len(y))
            ax = axes[i//2][i%2] if num>1 else axes[i]
            ax.scatter(x,z,c=y,s=10)
            ax.axis('equal')
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(name+'predict'))
        plt.close()

        fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,4*num))
        for i in range(n):
            i = target if target is not None else i
            s_off, s_on = s_st[0,:,i], s_st[1,:,i]
            ss, which = s_on, 0
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[measure.qubitToread[i]](d)
            percent1 = list(y).count(which)/len(y)
            measure.onwhich[measure.qubitToread[i]] = (which if percent1 > 0.5 else 1-which)
            measure.offwhich[measure.qubitToread[i]] = (1-which if percent1 > 0.5 else which)
            percent_on = list(y).count(measure.onwhich[measure.qubitToread[i]])/len(y)
            ax = axes[i//2][i%2] if num>1 else axes[i]
            ax.scatter(np.real(ss),np.imag(ss),c=y,s=10)
            ax.set_title(f'|1>pop={round(percent_on*100,3)}%')
            ax.axis('equal')
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(name+'e'))
        plt.close()

        fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,4*num))
        for i in range(n):
            i = target if target is not None else i
            s_off, s_on = s_st[0,:,i], s_st[1,:,i]
            ss, which = s_off, measure.offwhich[measure.qubitToread[i]]
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[measure.qubitToread[i]](d)
            percent_off = list(y).count(which)/len(y)
            measure.readmatrix[measure.qubitToread[i]] = np.mat([[percent_off,1-percent_on],[1-percent_off,percent_on]])
            ax = axes[i//2][i%2] if num>1 else axes[i]
            ax.scatter(np.real(ss),np.imag(ss),c=y,s=10)
            ax.set_title(f'|0>pop={round(percent_off*100,3)}%')
            ax.axis('equal')
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(name+'g'))
        plt.close()
    else:
        fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,4*num))
        for i in range(n):
            i = target if target is not None else i
            s_off, s_on = s_st[0,:,i], s_st[1,:,i]
            ss, which = s_on, measure.onwhich[measure.qubitToread[i]] 
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[measure.qubitToread[i]](d)
            percent_on = list(y).count(which)/len(y)
            ax = axes[i//2][i%2] if num>1 else axes[i]
            ax.scatter(np.real(ss),np.imag(ss),c=y,s=10)
            ax.set_title(f'|1>pop={round(percent_on*100,3)}%')
            ax.axis('equal')
        plt.savefig(r'\\QND-SERVER2\skzhao\fig\%s.png'%(name+'classify'))
        plt.close()
