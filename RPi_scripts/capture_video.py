#!/usr/bin/python3
import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

import os
import getpass

print("Env thinks the user is [%s]" % (os.getlogin()))
print("Effective user is [%s]" % (getpass.getuser()))

picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

encoder = H264Encoder(10000000)

picam2.start_recording(encoder, '/home/jakob/shared/video/test.h264')
time.sleep(5)
picam2.stop_recording()
