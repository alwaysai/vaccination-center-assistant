import os
import json
import sys
import time
import datetime
import requests

import edgeiq

def send_data(url, route, json):
    try:
        url = url + route
        requests.post(url=url, json=json)
    except:
        print("connection error, unable to send request")

class VaccineTracker():
    def __init__(self):
        self.id = "vaccination_area"
        self._send_events = False
        self.server_event_url = "http://localhost:5001/" # configure as needed
        self._start_time = time.time()

        # detection model
        self.detector = self.load_model("alwaysai/yolov3")

        self.centroid_tracker = edgeiq.CentroidTracker(deregister_frames=4, max_distance=130)
        
        # vaccination box
        self.box = edgeiq.BoundingBox(1269, 187, 1920, 1080) # configure as needed
        
        # for overall vaccination logic
        self.current_ids = []
        self.timestamp = None
        self.total_vaccinations = 0
        self.vaccination_time = 30
        self.scheduled_vaccinations = 20
        self.doses_per_vial = 10
        self.last_apt = datetime.datetime.today().replace(hour=16, minute=45)
        self.send_event(0)

    def has_events(self):
        return self._send_events

    def load_json(self, filepath):
        if os.path.exists(filepath) == False:
            raise Exception('File at {} does not exist'.format(filepath))
        with open(filepath) as data:
            return json.load(data)

    def load_model(self, model):
        # start up a first object detection model
        obj_detect = edgeiq.ObjectDetection(model)
        obj_detect.load(engine=edgeiq.Engine.DNN)

        # print the details of each model to the console
        print("Model:\n{}\n".format(obj_detect.model_id))
        print("Engine: {}".format(obj_detect.engine))
        print("Accelerator: {}\n".format(obj_detect.accelerator))
        print("Labels:\n{}\n".format(obj_detect.labels))

        return obj_detect	

    def has_expired(self):
        expired = self.timestamp is not None and \
            int(time.time()) >= self.timestamp + self.vaccination_time
        return expired
        
    def send_event(self, vaccinations=0):
        event_log = {}
        event_log['time_marker'] = str(round((time.time() - self._start_time), 2))
        vaccination_data = {}
        vaccination_data['new_vaccinations'] = vaccinations
        vaccination_data['total_vaccinations'] = self.total_vaccinations
        vaccination_data['vials_opened'] = self.calculate_vials_opened()
        vaccination_data['doses_left_in_current_vial'] = self.calculate_doses_in_current_vial()
        vaccination_data['appointments_remaining'] = self.scheduled_vaccinations - self.total_vaccinations
        vaccination_data['last_apt'] = str(self.last_apt)
        event_log['vaccination_data'] = vaccination_data
        print("event_log " + json.dumps(event_log, indent=4))
        send_data(self.server_event_url, "event", event_log)

    def calculate_vials_opened(self):
        if self.total_vaccinations == 0:
            return 0
        return 1 + int((self.total_vaccinations / self.doses_per_vial))
    
    def calculate_doses_in_current_vial(self):
        return self.doses_per_vial - (self.total_vaccinations % self.doses_per_vial)

    def check_overlap(self, people_predictions):
        in_area = []
        for key, prediction in people_predictions.items():
            if prediction.box.compute_overlap(self.box) > 0.99:
                in_area.append(key)
        return in_area

    def update(self, image):
        # if someone is in the chair -- we're waiting for a vaccine
        results = self.detector.detect_objects(image, confidence_level=0.6)
        people_pred = edgeiq.filter_predictions_by_label(results.predictions, ["person"])
        if len(people_pred) > 0:
    
            # now check how many people are in the vaccination areas
            predictions = self.centroid_tracker.update(people_pred)
            keys = self.check_overlap(predictions)
      
            if len(keys) == 2 and len(self.current_ids) == 0:
                # start tracking
                self.current_ids = keys
                self.timestamp = time.time()

            elif len(keys) <= 2:
                if self.has_expired():
                    self.total_vaccinations += 1
                    self.timestamp = None
                    self.current_ids = []
                    self.send_event(1)
