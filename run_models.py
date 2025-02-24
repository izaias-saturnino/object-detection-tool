import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import sys
import time
from object_resizer import resize_images
from train_model import read_model_metadata
from math import ceil, floor
from detection_post_processing import restore_broken_detection

def run_model(model_folder, model_name, confidence, save_results, images_dir, labels_dir, detection_path):
    model_metadata = read_model_metadata(model_folder, model_name)
    resize_stat_name = model_metadata["resize_stat_name"]
    resize_stat_value = model_metadata["resize_stat_value"]

    _, break_metadatas = resize_images(images_dir, labels_dir, detection_path, detection_path, resize_stat_name, resize_stat_value, borders=True)

    print("Model:", model_name)
    model_path = os.path.join(model_folder, model_name)
    model = YOLO(model_path)
    results = model.predict(detection_path, conf=confidence, save=save_results, augment=True, show_labels=False, mode="val", split="test")

    current_time = time.strftime("%Y%m%d-%H%M%S")

    images_dir_name = images_dir.split("/")[-1]

    try:
        run_dir = "runs_" + model_name + "_" + images_dir_name + "_" + current_time
        shutil.move("runs", run_dir)
    except:
        pass

    print("Post-processing detections...")
    results = restore_broken_detection(results, break_metadatas)

    return run_dir, results