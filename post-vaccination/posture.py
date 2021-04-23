# vaccination-app-suite/post-vaccination/posture.py
import time
import json
from collections import Counter
import sys
import os
import requests

import edgeiq

def send_data(url, route, json):
    try:
        url = url + route
        print(url)
        requests.post(url=url, json=json)
    except:
        print("connection error, unable to send request")

"""
Tracks current key_point coordinates and uses these to check for
a raised hand.
Stores a scale factor, which is used to reduce or increase the distances
that are used to calculate distance between key_points.
"""
class CheckPosture:

    def __init__(self, scale=1, key_points={}):
        self.key_points = key_points
        self.scale = scale
        self.message = ""
        self.interval = 3 # the time between the raised hand and the potential alert
        self.timestamp = None # marks when the raised hand was triggered
        self.listening = False
        self.signals = []
        self.people_count = 0
        self.previous_people_count = 0
        self.previous_and_raise_count = 0

        self._server_url = "http://localhost:5001/" # configure as needed
        self._start_time = time.time()

        self.pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")

        self.pose_estimator.load(
            engine=edgeiq.Engine.DNN)

        print("Loaded model:\n{}\n".format(self.pose_estimator.model_id))
        print("Engine: {}".format(self.pose_estimator.engine))
        print("Accelerator: {}\n".format(self.pose_estimator.accelerator))

    def is_listening(self):
        return self.listening

    def start_listening(self):
        self.timestamp = int(time.time())
        self.listening = True
        print("recieved start signal, aggregating data")

    def stop_listening(self):
        self.timestamp = None
        self.listening = False
        self.signals = []
        print("stopping listening until recieve start signal")
    
    def has_expired(self):
        return self.timestamp is not None and \
            int(time.time()) >= self.timestamp + self.interval  
    
    def set_key_points(self, key_points):
        """
        Updates the key_points dictionary used in calculations
        :param key_points: {}
            the dictionary to use for key_points
        """
        self.key_points = key_points

    def get_key_points(self):
        """
        Returns the instance's current version of the key_points dictionary
        :return: {}
            the current key_points dictionary
        """
        return self.key_points

    def set_people_count(self, count):
        self.previous_people_count = self.people_count
        self.people_count = count
    
    def get_people_count(self):
        return self.people_count

    def set_message(self, message):
        """
        Setter to update the message manually if needed
        :param message: string
            The message to override the current message
        """
        self.message = message

    def build_message(self):
        """
        Builds a string with advice to the user on how to correct their posture
        :return: string
            The string containing specific advice
        """
        current_message = ""
        if self.check_raised_hand():
            current_message += "[ALERT] Person needs assistance.\n"
        return current_message

    def get_message(self):
        """
        Getter method to return the current message
        :return: string
            The current posture message
        """
        return self.message

    def set_scale(self, scale):
        """
        Sets the scale factor to use for the posture calculations
        :param scale: int
            The value to scale the measurements used in the calculations by. Larger values will
            mean a less stringent calculation.
        """
        self.scale = scale

    def get_scale(self):
        """
        Returns the current scale for the instance
        :return: int
            The scale being used by the instance for posture calculations
        """
        return self.scale

    def check_raised_hand(self):
        """Checked whether a hand is raised.

        Returns:
            boolean: True if hand is raised.
        """
        # compare shoulder to elbow (make sure elbow is higher in the frame, smaller coordiate)
        if self.key_points['Left Shoulder'].y != -1 and self.key_points['Left Elbow'].y != -1 \
            and self.key_points['Left Shoulder'].y >= self.key_points['Left Elbow'].y \
            or self.key_points['Right Shoulder'].y != -1 and self.key_points['Right Elbow'].y != -1 \
            and self.key_points['Right Shoulder'].y >= self.key_points['Right Elbow'].y:
                print("hand is raised: testing shoulder vs elbow")            
                return True

        # add in other hand raise key point comparisons here

        return False
    
    def check_for_people_change(self):
        return self.people_count != self.previous_people_count

    def check_for_hand_raised(self):
        """
        Checks all current posture functions
        :return: Boolean
            True if all posture functions return True; False otherwise
        """
        hand_raised = self.check_raised_hand()
        self.signals.append(hand_raised)
        if self.is_listening():
            if self.has_expired():
                signals = [i for i in self.signals if i]

                # check if the number of positives each round against the number of people observed
                signal = True if len(signals) >= len(self.signals)/self.get_people_count() else False 
                self.stop_listening()
                return 1 if signal else 0
        elif hand_raised:

            # if we haven't encountered a hand raise recently
            # start listening if we encounter a hand raise
            self.start_listening()
            print("raised hand detected, initializing aggregator")
        return -1

    def send_events(self, hand_count, people_count):
        event_log = {}
        event_log['time_marker'] = str(round((time.time() - self._start_time), 2))
        event_log['hands_raised'] = hand_count
        event_log['post_vaccine_count'] = people_count
        print("event_log: {}".format(json.dumps(event_log)))
        
        # send alert to server
        send_data(self._server_url, "event", event_log)

    def update(self, frame):
        results = self.pose_estimator.estimate(frame)
        
        # Generate text to display on streamer
        text = ["Model: {}".format(self.pose_estimator.model_id)]
        text.append(
                "Inference time: {:1.3f} s".format(results.duration))

        hand_count = 0
        self.set_people_count(len(results.poses))

        for ind, pose in enumerate(results.poses):

            # update the instance key_points to check the posture
            self.set_key_points(pose.key_points)
            value = self.check_for_hand_raised()
            if value != -1:
                if value == 1:
                    hand_count += 1
                    print("person {} raising hand".format(ind))
                self.send_events(value, len(results.poses))

        frame = results.draw_poses(frame)
        text.append("{} people in total".format(len(results.poses)))
        text.append("{} people hands raised".format(hand_count))

        return frame, text