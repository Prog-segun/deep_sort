# File: TGI/Tracker/deep_sort/mc_deepsort.py

import sys
DEEPSORT_DIR = "Tracker/deep_sort"
sys.path.insert(0, DEEPSORT_DIR)

import numpy as np
import tensorflow as tf
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class MCDeepSort:
    def __init__(self, **kwargs):
        self.max_cosine_distance = 0.5
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        import os
        model_filename = os.path.join(DEEPSORT_DIR, 'model_data/mars-small128.pb')
        
        print(f"DeepSORT: Loading Re-ID model from '{os.path.abspath(model_filename)}'...")
        # --- CORRECTED: Unpack both the object and the function ---
        self.image_encoder_obj, self.encoder_func = gdet.create_box_encoder(model_filename, batch_size=1)
        
        print("DeepSORT: Re-ID model loaded.")
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric)
        print("✅ DeepSORT Tracker Initialized")

    def __call__(self, frame, detections, scores, class_ids):
        if detections is None or len(detections) == 0:
            self.tracker.predict(); self.tracker.update([]); return [], [], [], []
        
        xywh_bboxes = [ (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]) for bbox in detections ]
        
        # Call the correct encoder function
        features = self.encoder_func(frame, xywh_bboxes)
        
        deepsort_detections = [
            Detection(xywh_bboxes[i], scores[i], features[i], class_ids[i]) for i in range(len(xywh_bboxes))
        ]
        
        boxes = np.array([d.tlwh for d in deepsort_detections])
        scores_arr = np.array([d.confidence for d in deepsort_detections])

        # Use the correct object to access the session
        indices = self.image_encoder_obj.session.run(tf.image.non_max_suppression(
            boxes, scores_arr, max_output_size=len(boxes), iou_threshold=self.nms_max_overlap
            ))
        
        deepsort_detections = [deepsort_detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(deepsort_detections)
        
        track_ids, bboxes, final_scores, final_class_ids = [], [], [], []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            track_ids.append(track.track_id)
            bboxes.append(track.to_tlbr())
            final_scores.append(track.get_latest_score())
            final_class_ids.append(track.get_latest_class())
        return np.array(track_ids), np.array(bboxes), np.array(final_scores), np.array(final_class_ids)

from deep_sort.track import Track
def get_latest_score(self): return self.last_score
def get_latest_class(self): return self.class_id
Track.get_latest_score = get_latest_score
Track.get_latest_class = get_latest_class