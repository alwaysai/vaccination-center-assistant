import time
import sys
import os

import cv2
import edgeiq
from detection_manager import DetectionManager

def main():

    dm = DetectionManager()

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video, \
                edgeiq.Streamer() as streamer:
            time.sleep(2.0)
            fps.start()

            while True:                  
                image = video.read()            
                (image, text) = dm.update(image)
                streamer.send_data(image, text)
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