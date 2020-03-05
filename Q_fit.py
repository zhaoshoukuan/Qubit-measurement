import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import least_squares as ls 

def preFit(f, s21):
        A = np.poly1d(np.polyfit(f, np.abs(s21), 1))
        phi = np.unwrap(np.angle(s21), 0.9 * np.pi)
        phase = np.poly1d(np.polyfit(f, phi, 1))
        s21 = s21 / A(f) / np.exp(1j * phase(f))
        return f, s21

def circleLeastFit(x, y):
        def circle_err(params, x, y):
            xc, yc, R = params
            return (x - xc)**2 + (y - yc)**2 - R**2

        p0 = [
            x.mean(),
            y.mean(),
            np.sqrt(((x - x.mean())**2 + (y - y.mean())**2).mean())
        ]
        res = ls(circle_err, p0, args=(x, y))
        return res.x

def guessParams(x, s):
    
    y = np.abs(1 / s)
    f0 = x[y.argmax()]
    _bw = x[y > 0.5 * (y.max() + y.min())]
    FWHM = max(_bw) - min(_bw)
    Qi = f0 / FWHM
    _, _, R = circleLeastFit(np.real(1 / s), np.imag(1 / s))
    Qe = Qi / (2 * R)
    QL = 1 / (1 / Qi + 1 / Qe)

    return [f0, Qi, Qe, QL, 0, 1, 0, 0, 0, 0]

def invS21(f, f0, Qi, Qe, phi):
        #QL = 1/(1/Qi+1/Qe)
        return 1 + (Qi / np.abs(Qe) * np.exp(1j * phi)) / (
            1 + 2j * Qi * (np.abs(f) / np.abs(f0) - 1))

def fit(x, s,
                params=None,
                with_delay=False,
                with_high_order=False):

        def err(params,
                f,
                s21,
                with_delay=with_delay,
                with_high_order=with_high_order):
            f0, Qi, Qe, phi, A, Aphi, delay, a, b = params
            background = A * (with_high_order *
                              (a * (f - f0)**2 + b *
                               (f - f0)) + 1) * np.exp(1j *
                                                       (with_delay * delay *
                                                        (f - f0) + Aphi))
            y = 1 / s21 - invS21(f, f0, Qi, Qe, phi) / background
            return np.abs(y)

        if params is None:
            f0, Qi, Qe, QL, phi, A, Aphi, delay, a, b = guessParams(
                x, s)
        else:
            f0, Qi, Qe, QL, phi, A, Aphi, delay, a, b = params

        res = ls(err, [f0, Qi, Qe, phi, A, Aphi, delay, a, b], args=(x, s))
        f0, Qi, Qe, phi, A, Aphi, delay, a, b = res.x
        QL = 1 / (1 / Qi + 1 / Qe)
        return f0, Qi, Qe, QL, phi, A, Aphi, delay, a, b

if __name__=='__main__':
    data = np.loadtxt('C:\\Users\\赵寿宽\\Desktop\\桌面资料\\VScode\\python\\MyModule\\cav.txt')
    f, s21_r, s21_im = data[:,0], data[:,1], data[:,2]
    s21_old = s21_r + 1j*s21_im
    f, s21 = preFit(f,s21_old)
    para = fit(f,s21)
    pa = para[0],para[1],para[2],para[4]
    xc, yc, R = circleLeastFit(np.real(1 / s21), np.imag(1 / s21))
    
    #画图
    fig = plt.figure(figsize=(14,10))
    ax1 = plt.subplot(221)
    ax1.plot(f,np.abs(s21),'r.')
    ax1.plot(f,1/np.abs(invS21(f,*pa)),'b-')
    ax1.set_title('f0:%f,Qi:%f'%(para[0]/1e9,para[1]))

    ax2 = plt.subplot(222)
    theta = np.linspace(0,2*np.pi,1000)
    x = xc + R*np.cos(theta)
    y = yc + R*np.sin(theta)
    ax2.plot((1/s21).real,(1/s21).imag,'.')
    ax2.plot(x,y,'-')
    ax2.axis('equal')

    ax3 = plt.subplot(223)
    #ax3.plot(f,np.angle(s21_old),'r.')
    ax3.plot(f,np.abs(s21))

    ax4 = plt.subplot(224)
    ax4.plot(f,np.angle(s21),'r.')
    plt.show()
