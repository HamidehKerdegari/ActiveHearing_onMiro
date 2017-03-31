#!/usr/bin/env python

#######################################################################################################################
import time
import os, thread
from multiprocessing import Process

from lib import miro_tcp, miro_record


## MAIN ##
######################################################################################################################
if __name__ == '__main__':
    client = miro_tcp.client('192.168.1.2', 1234)

    while True:
        c = 0
        for Neck in ['000', '050', '100']: # Lift motion
            for Head in ['000', '050', '100']: # left right motion
                Right_e = '000'
                Left_e = '000'
                audio_path = 'recordings/N'+Neck + 'H'+Head + 'R'+Right_e + 'L'+Left_e + '.wav'
                img_path = 'recordings/N'+Neck + 'H'+Head + 'R'+Right_e + 'L'+Left_e + '.jpg'
                if not os.path.exists(audio_path):
                    c+=1
                    client.cmd = 'Joint_Pos ' + Left_e + ' ' + Right_e + ' ' + Head + ' ' + Neck
                    time.sleep(2) # Wait for Miro to move into the new position...
                    os.system("imagesnap -w 1 -d 'Microsoft LifeCam VX-5000' " + img_path)

                    p1=Process(target = miro_record.play, args=('lib/sweep.wav',))
                    p2=Process(target = miro_record.record, args=(audio_path,))
                    p1.start()
                    p2.start()
                    time.sleep(13) # Wait for recording to finish...
        if c==0:
            break

    print '***** End Of Recordings *****'
######################################################################################################################
