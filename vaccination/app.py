#vaccination-app-suite/vaccination/app.py

import logging
import time
import os
import json
import numpy as np
import sys

import cv2
import edgeiq
from vaccine_tracker import VaccineTracker

"""
Modifies realtime_object_detector to detect whether a person is exhibiting
improper posture (as defined in CheckPosture).

Proper posture functions are defined in the 'CheckPosture' class and imported.

This app uses a configuration file (config.json) to configure a
'scale' factor, which is used to make the posture calculations more or less stringent.
A larger scale factor (>1) makes the calculation less stringent (allows for less straight
posture).
"""


def main():
    fps = edgeiq.FPS()

    try:
            streamer = edgeiq.Streamer()
            streamer.setup()
            video_stream = edgeiq.WebcamVideoStream(cam=0) # replace with FileVideoStream if need be
       
            # Allow application to warm up
            video_stream.start()
            time.sleep(3.0)
            fps.start()
            text =[""]

            # initialize Vaccine Trakcer
            vaccine_tracker = VaccineTracker()

            # loop detection
            while True:
                frame = video_stream.read()
                vaccine_tracker.update(frame)

                # draw the vaccination box in the frame
                frame = edgeiq.markup_image(frame, [edgeiq.ObjectDetectionPrediction(label="vaccination", index=0, box=vaccine_tracker.box, confidence=100.00)])
                streamer.send_data(frame, text)
                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        streamer.close()
        video_stream.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
