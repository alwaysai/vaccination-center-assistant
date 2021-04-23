#vaccination-app-suite/posture-corrector/app.py

import logging
import time
import os
import json
import numpy as np

from posture import CheckPosture
import edgeiq

"""
Modifies realtime_pose_estimator to detect whether a person is exhibiting
a raised hand (as defined in CheckPosture).
"""

def main():

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            check_posture = CheckPosture()

            # loop detection
            while True:
                frame = video_stream.read()
                frame, text = check_posture.update(frame)
                streamer.send_data(frame, text)
                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
