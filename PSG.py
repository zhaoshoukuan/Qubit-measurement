import numpy as np

from qulab import BaseDriver, QList, QOption, QReal


class Driver(BaseDriver):
    support_models = ['E8257D', 'SMF100A', 'SMB100A', 'SGS100A']

    quants = [
        QReal('Frequency',
              unit='Hz',
              set_cmd=':FREQ %(value).13e%(unit)s',
              get_cmd=':FREQ?'),
        QReal('Power',
              unit='dBm',
              set_cmd=':POWER %(value).8e%(unit)s',
              get_cmd=':POWER?'),
        QOption('Output',
                set_cmd=':OUTP %(option)s',
                get_cmd=':OUTP?',
                options=[('OFF', 'OFF'), ('ON', 'ON')]),
        QOption('Moutput',
                set_cmd='PULM:STAT %(option)s',
                get_cmd='PULM:STAT?',
                options=[('OFF', 'OFF'), ('ON', 'ON')]),
        QOption('Mform',
                set_cmd='PULM:SOUR %(option)s',
                get_cmd='PULM:SOUR?',
                options=[('EXT', 'EXT'), ('INT', 'INT')]),   
    ]
