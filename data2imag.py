from qulab.storage import connect
from qulab.storage.schema import Record, base
import matplotlib.pyplot as plt
import numpy as np, imp
import qulab.optimize
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


def write(phase=None,addr='mongodb',height=2,title=None,which=0,peak=110,dB=False,dePhase=False):
    if addr == 'mongodb':
        data, ID, comment, tags, name, finishtime = read(title=title,which=which)
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
                if dePhase :
                    rows, s = op.RowToRipe().deductPhase(rows,s)

                rows = rows / 1e9 if rows[0] / 1e9 > 1 else rows 
                cols = cols / 1e9 if cols[0] / 1e9 > 1 else cols 
                v.append((cols,rows,s))
                extent = [min(rows),max(rows),min(cols),max(cols)]
                res = 20*np.log10(np.abs(s)) if dB else np.abs(s)
                if n != 1:
                    im0 = axes[i//2][i%2].imshow(res,extent=extent,aspect='auto',origin='lower',interpolation='nearest',cmap='jet')
                    im1 = axes[i][1].imshow(np.angle(s),extent=extent,aspect='auto',origin='lower',interpolation='nearest')
                    plt.colorbar(im0,ax=axes[i//2][i%2])
                    plt.colorbar(im1,ax=axes[i][1])
                else:
                    im0 = axes[0].imshow(res,extent=extent,aspect='auto',origin='lower',interpolation='nearest')
                    im1 = axes[1].imshow(np.angle(s),extent=extent,aspect='auto',origin='lower',interpolation='nearest',cmap='jet')
                    plt.colorbar(im0,ax=axes[0])
                    plt.colorbar(im1,ax=axes[1])
        else:
            n = num // 2 + num % 2
            fig, axes = plt.subplots(ncols=2,nrows=n,figsize=(9,height*n))
            fig.subplots_adjust(top=0.9,bottom=0.1,hspace=0.5)
            v = []
            for i in range(num): 
                cols, rows, s= data[0][:,i], data[1][0,:,i], data[2][:,:,i]
                rows = rows / 1e9 if rows[0] / 1e9 > 1 else rows 
                cols = cols / 1e9 if cols[0] / 1e9 > 1 else cols 
                # s[np.abs(s) > peak] = s[np.abs(s)==np.min(np.abs(s))]
                v.append((cols,rows,s))
                extent = [min(rows),max(rows),min(cols),max(cols)]
                res = 20*np.log10(np.abs(s)) if dB else np.abs(s)
                if n != 1:
                    im = axes[i//2][i%2].imshow(res,extent=extent,aspect='auto',origin='lower',\
                                                interpolation='nearest',cmap='jet')
                    axes[i//2][i%2].set_title(tags[0])
                    plt.colorbar(im,ax=axes[i//2][i%2])
                   
                else:
                    im = axes[i].imshow(res,extent=extent,aspect='auto',origin='lower',interpolation='nearest',cmap='jet')
                    axes[i].set_title(tags[0])
                    plt.colorbar(im,ax=axes[i])
        return v,num,tags,finishtime