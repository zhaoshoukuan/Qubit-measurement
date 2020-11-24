from qulab.storage import connect
from qulab.storage.schema import Record, base
import matplotlib.pyplot as plt
import numpy as np, imp
import qulab.optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
op = imp.reload(qulab.optimize)

connect.connect()

def read(title=None,which=0):
    
    if title is None:
        x = Record.objects.order_by('-finished_time')[which]
    else:
        x = Record.objects(title=title).order_by('-finished_time')[which]
    tags, comment, name, finishtime = x['tags'], x['comment'], x['title'], x['finished_time']
    data, ID = base.from_pickle(x.datafield), x.id
    return data, ID, comment, tags, name, finishtime


def write(phase=None,addr='mongodb',height=2,title=None,which=0,peak=110,dB=False,dePhase=False,transposition=False):
    if addr == 'mongodb':
        data, ID, comment, tags, name, finishtime = read(title=title,which=which)
        if len(data) == 3:
            if len(data[2][-1]) != len(data[2][0]):
                print(1)
                cols, rows, s = data[0][:-1], data[1][:-1], data[2][:-1]
                data = (cols,rows,s)
    else:
        d = np.load(addr)
        data, tags = (d['row'],d['col'],d['s']), d['tags']

    if len(data) == 2:
        num = len(data[0].T)
        if phase == 'phase': 
            fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,height*num))
            fig.subplots_adjust(top=0.9,bottom=0.1,hspace=0.5)
            v = []
            for i in range(num): 
                f, s = data[0][:,i],data[1][:,i]
                f = f / 1e9 if f[0] / 1e9 > 1 else f
                v.append((f,s))
                res = 20*np.log10(np.abs(s)) if dB else np.abs(s)
                if num != 1:
                    
                    axes[i][0].plot(f,res)
                    axes[i][1].plot(f,np.angle(s))
                else:
                    axes[0].plot(f,res)
                    axes[1].plot(f,np.angle(s))
        else:
            n = num // 2 + num % 2
            fig, axes = plt.subplots(ncols=2,nrows=n,figsize=(9,height*n))
            fig.subplots_adjust(top=0.85,bottom=0.15,hspace=0.5)
            v = []
            for i in range(num): 
                f, s = data[0][:,i],data[1][:,i]
                f = f / 1e9 if f[0] / 1e9 > 1 else f
                v.append((f,s))
                res = 20*np.log10(np.abs(s)) if dB else np.abs(s)
                if n != 1:
                    axes[i//2][i%2].plot(f,res,'-.')
                    #axes[i//2][i%2].set_title(name)
                else:
                    axes[i].plot(f,res)
                    if addr == 'mongodb':
                        axes[i].set_title(name)
        return v,num,tags,finishtime
    if len(data) == 3 :
        
        num = len(data[0].T)
        if phase == 'phase': 
            n = num // 2 + num % 2
            fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,height*num))
            fig.subplots_adjust(top=0.9,bottom=0.1,hspace=0.5)
            v = []
            for i in range(num): 
                cols, rows, s= data[0][:,i], data[1][0,:,i], data[2][:,:,i]

                # if transposition:
                #     s = s.T
                if dePhase :
                    rows, s = op.RowToRipe().deductPhase(rows,s)

                rows = rows / 1e9 if rows[0] / 1e9 > 1 else rows 
                cols = cols / 1e9 if cols[0] / 1e9 > 1 else cols 
                v.append((cols,rows,s))
                extent = [min(rows),max(rows),min(cols),max(cols)]
                res = 20*np.log10(np.abs(s)) if dB else np.abs(s)
                cols_,rows_ = np.meshgrid(rows,cols)
                if transposition:
                    s = s.T
                    cols_,rows_ = np.meshgrid(cols,rows)

                if n != 1:
                    # im0 = axes[i//2][i%2].imshow(res,extent=extent,aspect='auto',origin='lower',interpolation='nearest',cmap='jet')
                    # im1 = axes[i][1].imshow(np.angle(s),extent=extent,aspect='auto',origin='lower',interpolation='nearest')
                    im0 = axes[i//2][0].pcolormesh(cols_,rows_,np.abs(res),cmap='jet')
                    im1 = axes[i//2][1].pcolormesh(cols_,rows_,np.angle(res),cmap='jet')
                    plt.colorbar(im0,ax=axes[i//2][0])
                    plt.colorbar(im1,ax=axes[i//2][1])
                else:
                    # im0 = axes[0].imshow(res,extent=extent,aspect='auto',origin='lower',interpolation='nearest')
                    # im1 = axes[1].imshow(np.angle(s),extent=extent,aspect='auto',origin='lower',interpolation='nearest',cmap='jet')
                    im0 = axes[0].pcolormesh(cols_,rows_,np.abs(s),cmap='jet')
                    im1 = axes[1].pcolormesh(cols_,rows_,np.angle(s),cmap='jet')
                    plt.colorbar(im0,ax=axes[0])
                    plt.colorbar(im1,ax=axes[1])
        else:
            n = num // 2 + num % 2
            fig, axes = plt.subplots(ncols=2,nrows=n,figsize=(9,height*n))
            fig.subplots_adjust(top=0.9,bottom=0.1,hspace=0.5)
            v = []
            for i in range(num): 
                cols, rows, s= data[0][:,i], data[1][0,:,i], data[2][:,:,i]
                # if transposition:
                #     s = s.T
                rows = rows / 1e9 if rows[0] / 1e9 > 1 else rows 
                cols = cols / 1e9 if cols[0] / 1e9 > 1 else cols 
                # s[np.abs(s) > peak] = s[np.abs(s)==np.min(np.abs(s))]
                v.append((cols,rows,s))
                extent = [min(rows),max(rows),min(cols),max(cols)]
                cols_,rows_ = np.meshgrid(rows,cols)
                if transposition:
                    s = s.T
                    cols_,rows_ = np.meshgrid(cols,rows)
                res = 20*np.log10(np.abs(s)) if dB else np.abs(s)
                if n != 1:
                    # im = axes[i//2][i%2].imshow(res,extent=extent,aspect='auto',origin='lower',\
                    #                             interpolation='nearest',cmap='jet')
                    im = axes[i//2][i%2].pcolormesh(cols_,rows_,np.abs(res),cmap='jet')
                    axes[i//2][i%2].set_title(tags[0])
                    plt.colorbar(im,ax=axes[i//2][i%2])
                   
                else:
                    # im = axes[i].imshow(res,extent=extent,aspect='auto',origin='lower',interpolation='nearest',cmap='jet')
                    im = axes[i].pcolormesh(cols_,rows_,np.abs(res),cmap='jet')
                    axes[i].set_title(tags[0])
                    plt.colorbar(im,ax=axes[i])
        return v,num,tags,finishtime

def slidePlot(fig,ax,var):
    
    a = 0
    b = 1
    f = a + b*var
    l, = plt.plot(var, f,'k', lw=2)

    ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    axa = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axb = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sa = Slider(axa, 'a', -1, 1, valinit=a,valstep=0.0005)
    sb = Slider(axb, 'b', -1000, 1000, valinit=b,valstep=0.0005)


    def update(val):
        bi = sb.val
        ai = sa.val
        l.set_ydata(ai + bi*var)
        fig.canvas.draw_idle()


    sa.on_changed(update)
    sb.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        sa.reset()
        sb.reset()
    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    radio.on_clicked(colorfunc)

    # Initialize plot with correct initial active value
    colorfunc(radio.value_selected)
    plt.show()
    return sa,sb