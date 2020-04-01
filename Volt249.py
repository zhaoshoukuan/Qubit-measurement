from qulab.VoltageSettingCore import *

SetDefaultIP("10.122.7.249")
SetVolMax(9.8)
SetVolMin(-9.8)

def DC(ch,volt):

    # Set for Channel 1:
    SetChannelNum(ch,0)
    voltageValue = volt
    dValue = CalculateDValue(voltageValue)
    SetDValue(dValue)

