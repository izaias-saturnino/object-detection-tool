import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import sys
import time

model_folder = "models"
model_names = ["yolov8n_black_particles-obb_phase_1.pt", "yolo11n-obb_test.pt_20250218-072546.pt"]
confidence = 0.5
save_results = True
# dir_paths = ["datasets/data_TEM/images/train", "datasets/data_TEM/images/val"]
dir_paths = ["to_execute"]

def run_model(model_folder, model_name, confidence, save_results, dir_paths, save_csv=True, write_ids=True):
    run_dirs = []
    for dir_path in dir_paths:
        images_path = dir_path
        print("-" * 50)
        print("Model:", model_name)
        model_path = os.path.join(model_folder, model_name)
        model = YOLO(model_path)
        results = model.predict(images_path, conf=confidence, save=save_results, augment=True, show_labels=False, mode="val", split="test")

        current_time = time.strftime("%Y%m%d-%H%M%S")

        dir_path_name = dir_path.split("/")[-1]

        try:
            run_dir = "runs_" + model_name + "_" + dir_path_name + "_" + current_time
            shutil.move("runs", run_dir)
            run_dirs.append(run_dir)
        except:
            pass
        print("-" * 50)

        if save_csv:
            for result in results:
                pass # TODO: save results to csv alongside the image names (write ids if write_ids is True)

    return run_dirs

def run_models(model_folder, model_names, confidence, save_results, dir_paths, save_csv=True, write_ids=True):
    run_dirs = []
    for model_name in model_names:
        model_run_dirs = run_model(model_folder, model_name, confidence, save_results, dir_paths, save_csv=save_csv, write_ids=write_ids)
        run_dirs.extend(model_run_dirs)
    return run_dirs

if __name__ == "__main__":
    run_models(model_folder, model_names, confidence, save_results, dir_paths)