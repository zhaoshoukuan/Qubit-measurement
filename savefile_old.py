#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   savefile.py
@Time    :   2019/03/03 11:23:28
@Author  :   sk zhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib


import numpy as np 
import pandas as pd 
from datetime import datetime
import os
import time
from pathlib import Path

class SaveFile():

    def __init__(self,tags1,tags2,*value):
        self.value = value
        self.tags1 = tags1
        self.tags2 = tags2
        #self.now = time.strftime("%H %M %S", time.localtime()) + '.txt'
        self.now = self.tags2 + '.txt'
        self.l = self.tags2 + '.lock'
        self.t = time.strftime('%y%m%d')
        self.p = Path('d:/skzhao/Data') / self.t / self.tags1 

    def SaveNow(self):
        
        if os.path.exists(self.p) == 0:
            os.makedirs(self.p)

        with open(self.p / self.l,'w') as f:
            f.write('1')
        v = pd.DataFrame(self.value[2],index=self.value[0],columns=self.value[1])
        for i in range(10):
            try:
                v.to_csv(self.p / self.now, sep=' ')
                break
            except PermissionError:
                pass
        os.remove(self.p / self.l)
        return 'd:/skzhao/Data' + '/' + self.t + '/' + self.tags1 + '/' + self.tags2
