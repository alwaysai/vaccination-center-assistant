import time
import itertools
from math import sqrt
from copy import deepcopy
import os
import json
import requests

import numpy as np
import edgeiq

START_TIME = time.time()

def send_data(url, route, json):
    try:
        url = url + route
        requests.post(url=url, json=json)
    except:
        print("connection error, unable to send request")

class InterestItem:
    """This class is used to calculate the distance scale.
    """
    def __init__(self, width, height, name, interest_labels):
        self.width = width
        self.height = height
        self.name = name
        self.interest_labels = interest_labels
    
    def __repr__(self):
        return "{} {} {} {}".format(self.interest_labels, self.width, self.height, self.name)

    def get_area(self):
        return self.width * self.height

class DetectionManager:
    def __init__(self):
        # client configuration
        self.id = "waiting_room"
        self.box = edgeiq.BoundingBox(0, 0, 1920, 1080)
        self.server_event_url = "http://localhost:5001/" # configure as needed
        self.chairs = { # configure as needed
            # 'chair1': 1
        }
        self.capacity = 4 # configure as needed

        # detection models
        self.detector = self.load_model("alwaysai/yolov3")
        self.mask_detector = self.load_model("<username>/<model_name>") # train and use your model here!

        # labels of interest (used to filter predictions)
        self.interest_items = {
            "person": InterestItem(16, 42, "person", ["person"])
        }

        # tracker to associate object ids with predictions
        self.centroid_tracker = edgeiq.CentroidTracker(deregister_frames=4, max_distance=130)

        self.covid_event_log = {}
        self.event_log = {}
        self.send_setup()
    
    def load_json(self, filepath):
        if os.path.exists(filepath) == False:
            raise Exception('File at {} does not exist'.format(filepath))
        with open(filepath) as data:
            return json.load(data)

    def send_setup(self):
        setup = {}
        setup['device_id'] = self.id
        setup['area'] = self.capacity
        setup['chairs'] = self.chairs
        print("[INFO] sending set up " + str(setup))
        send_data(self.server_event_url, "setup", setup)
    
    def load_model(self, model):
        # start up a first object detection model
        obj_detect = edgeiq.ObjectDetection(model)
        obj_detect.load(engine=edgeiq.Engine.DNN, accelerator=edgeiq.Accelerator.CPU)

        # print the details of each model to the console
        print("Model:\n{}\n".format(obj_detect.model_id))
        print("Engine: {}".format(obj_detect.engine))
        print("Accelerator: {}\n".format(obj_detect.accelerator))
        print("Labels:\n{}\n".format(obj_detect.labels))

        return obj_detect	
    
    def get_mask_results(self, predictions, image):
        """Searches each prediction box section of the input image for a mask,
        and generates a new prediction in the overall image based on the prediction.

        Args:
            predictions (list): List of ObjectDetectionPredictions
            image (numpy array): The image to inference on

        Returns:
            list: Returns a list of ObjectDetectionPrediction elements
        """
        mask_results = []
        for prediction in deepcopy(predictions):

            # use the person prediction to narrow the focus and search for masks
            new_image = edgeiq.cutout_image(image, prediction.box)
            mask_predictions = self.mask_detector.detect_objects(new_image, confidence_level=0.2).predictions

            if len(mask_predictions) > 0:
                pred = mask_predictions[0]

                # update the label with the mask model's label if it is found
                prediction.label = pred.label

                # make the new box in the original frame
                new_start_x = prediction.box.start_x + pred.box.start_x
                new_end_x = new_start_x + pred.box.width
                new_start_y = prediction.box.start_y + pred.box.start_y
                new_end_y = new_start_y + pred.box.height

                # send back this new box to be marked up in the frame
                new_box = edgeiq.BoundingBox(new_start_x, new_start_y, new_end_x, new_end_y)
                prediction.box = new_box
            else:
                prediction.label = "no-mask-detected"

            mask_results.append(prediction)
        return mask_results

    def get_distances(self, predictions):
        """Computes the distance between each pair of predictions, updates
        the event_log with additional data, and returns lists of people who
        are distanced and people who are not.

        Args:
            predictions (ObjectDetectionPrediction): A list of predictions

        Returns:
            (list, list): Returns a list of people who are distanced 
            and a list of people who are not
        """
        badkeys = set()
        goodlist, badlist, pairlist = {}, {}, {}
        pixel_scale, sum_dist, count = 0, 0, 0
        pairs = itertools.combinations(predictions.keys(), 2)
        for pair in pairs:
            count += 1
            
            # calculate scale as pixels per inch
            p1 = predictions.get(pair[0])
            p2 = predictions.get(pair[1])
            pixel_scale = self.get_pixel_scale([p1, p2])
            
            if pixel_scale > 0:
                # calculate distance in inches by dividing pixels by pixels per inch
                dist = p1.box.compute_distance(p2.box)/pixel_scale
                sum_dist += dist
                pairlist['{}-{}'.format(pair[0], pair[1])] = dist
                if dist < 42: # configure as needed
                    if pair[0] not in badkeys:
                        badlist[pair[0]] = p1
                    
                    if pair[1] not in badkeys:
                        badlist[pair[1]] = p2
                    badkeys.add(pair[0])
                    badkeys.add(pair[1])
        
        for key in predictions.keys():
            if key not in badkeys:
                goodlist[key] = predictions[key]
        
        if count > 0:
            ave_distance = sum_dist/count
        else:
            ave_distance = 0
        self.covid_event_log['distances'] = pairlist
        self.covid_event_log['ave_distance'] = ave_distance
        
        return goodlist, badlist

    def get_pixel_scale(self, predictions):
        """Calculates the appropriate scale for the passed in predication.

        Args:
            predictions (ObjectDetectionPrediction): List of predictions

        Returns:
            int: The average scale to used based on the predictions
        """
        pixel_scales = []
        for pred in predictions:
            item = self.interest_items.get(pred.label)
            if item is not None:
                pixel_scales.append(sqrt(pred.box.area)/sqrt(item.get_area()))

        if len(pixel_scales) > 0:
            return sum(pixel_scales)/len(pixel_scales)
        return 0
    
    def update(self, image):
        """Performs mask detection and distance calculation, marks up the image
        with appropriate colored boxes, checks for new events and sends alerts, and
        returns a text update and new image for the calling function to use. 

        Args:
            image (numpy array): The image to inference on

        Returns:
            (image, text): Returns the marked up image and text of the application status
        """
        self.covid_event_log = {}
        goodlist, badlist, mask_pred, no_mask_pred = [], [], [], []
        text = []
        self.covid_event_log['people_not_distanced'] = []
        self.covid_event_log['people_distanced'] = []
        self.covid_event_log['no_masks'] = 0
        self.covid_event_log['masks'] = 0
        self.covid_event_log['uncertain_masks'] = 0

        # get predictions as regular
        results = self.detector.detect_objects(image, confidence_level=0.99)
        
        # filter by labels of interest (i.e. 'person')
        people_pred = edgeiq.filter_predictions_by_label(results.predictions, list(self.interest_items.keys()))
        if len(people_pred) > 0:
        
            # send them to the centroid_tracker tracker
            # now we have results in format: {object_id: ObjectDetectionPrediction}
            tracked_people_pred = self.centroid_tracker.update(people_pred)

            # get area update
            keys = self.check_overlap(tracked_people_pred)

            new_predictions = {}
            for object_id, prediction in tracked_people_pred.items():
                if object_id in keys:
                    new_predictions[object_id] = prediction
        
            # map tracked objects to distance detection
            good_dist, bad_dist = self.get_distances(new_predictions)

            goodlist.extend(list(good_dist.values()))
            badlist.extend(list(bad_dist.values()))

            self.covid_event_log['people_not_distanced'] = list(bad_dist.keys())
            self.covid_event_log['people_distanced'] = list(good_dist.keys())
            text.append("{} people not distanced\n".format(len(bad_dist)))

        # map tracked objects to mask_detection
        mask_predictions = self.get_mask_results(people_pred, image)
        #print("mask_predictions {}".format(mask_predictions))

        if len(mask_predictions) > 0: 
            # get people predictions, to detect uncertain masks
            no_mask_pred, mask_pred, uncertain = self.map_mask_predictions(mask_predictions)

            self.covid_event_log['no_masks'] = len(no_mask_pred)
            self.covid_event_log['masks'] = len(mask_pred)
            self.covid_event_log['uncertain_masks'] = len(uncertain)
            no_mask_pred.extend(uncertain)

            text.append("{} people not wearing masks".format(len(no_mask_pred)))
            
        goodlist.extend(mask_pred)
        badlist.extend(no_mask_pred)

        image = edgeiq.markup_image(
                        image, badlist, show_labels=True, line_thickness=2, font_size=2, font_thickness=3, show_confidences=False, colors=[(0,0,255)])
        
        image = edgeiq.markup_image(
                        image, goodlist, show_labels=True, line_thickness=2, font_size=2, font_thickness=3, show_confidences=False, colors=[(12,105,7)])

        # send any relevant results to the server
        self.check_for_events()
        
        return image, text

    def check_overlap(self, people_predictions):
        """Checks for overlap of each person in people_predictions with the
        overall application box. You can add checks for chairs as well.

        Args:
            people_predictions (list): All of the ObjectDetectionPrediction elements

        Returns:
            list: List of IDs of people in the area
        """
        in_area = []
        for key, prediction in people_predictions.items():
            if prediction.box.compute_overlap(self.box) > 0.70: # configure as needed
                in_area.append(key)
        self.event_log['in_area'] = in_area
        return in_area

    def map_mask_predictions(self, mask_results):
        """Splits masks detections into three categories: wearing a mask, 
        not wearing one, or unknown.

        Args:
            mask_results (list): The list of ObjectDetectionPrediction elements for mask detection

        Returns:
            (list, list, list): Returns three lists
        """
        bad_masks, good_masks, uncertain = [], [], []
        
        for result in mask_results:
            if result.label == 'mask':
                good_masks.append(result)
            elif result.label == 'no-mask':
                bad_masks.append(result)
            else:
                uncertain.append(result)
                
        return bad_masks, good_masks, uncertain
    
    def check_for_events(self):
        if len(self.event_log) > 0:
            event_log = deepcopy(self.event_log)
            event_log['device_id'] = self.id
            event_log['time_marker'] = str(round((time.time() - START_TIME), 2))
            event_log['covid_data'] = self.covid_event_log
            print("event_log " + json.dumps(event_log, indent=4))
            send_data(self.server_event_url, "event", event_log)
