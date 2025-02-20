import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import sys
import time
from object_resizer import resize_images
from train_model import read_model_metadata

def restore_broken_detection(results, break_metadatas):
    print("Not implemented.")
    return results

def run_model(model_folder, model_name, confidence, save_results, images_dir, labels_dir, temp_data):
    model_metadata = read_model_metadata(model_folder, model_name)
    resize_stat_name = model_metadata["resize_stat_name"]
    resize_stat_value = model_metadata["resize_stat_value"]

    _, break_metadatas = resize_images(images_dir, labels_dir, temp_data, temp_data, resize_stat_name, resize_stat_value, borders=True)

    print("-" * 50)
    print("Model:", model_name)
    model_path = os.path.join(model_folder, model_name)
    model = YOLO(model_path)
    results = model.predict(temp_data, conf=confidence, save=save_results, augment=True, show_labels=False, mode="val", split="test")

    try:
        shutil.rmtree(temp_data)
    except:
        pass

    current_time = time.strftime("%Y%m%d-%H%M%S")

    images_dir_name = images_dir.split("/")[-1]

    try:
        run_dir = "runs_" + model_name + "_" + images_dir_name + "_" + current_time
        shutil.move("runs", run_dir)
    except:
        pass
    print("-" * 50)

    # TODO: restore broken detection
    results = restore_broken_detection(results, break_metadatas)

    return run_dir, results

if __name__ == "__main__":
    model_run_dirs, results_array_model = run_model("models", "yolo11n-obb_test.pt_20250218-072546.pt", 0.5, True, "to_execute", "to_execute")
    print("Results:")
    print(results_array_model)
    print("Run directories:")
    print(model_run_dirs)
    print("Done.")