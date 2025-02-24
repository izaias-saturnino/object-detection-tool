import shutil
import time
from train_model import train_model
import shutil
import time
import numpy as np
import os
import cv2
import threading
import multiprocessing
from config import load_config, write_config
from image_pre_processing import pre_process_images, generate_labels
from run_models import run_model
from train_model import train_model
from detection_post_processing import save_obbs_to_csv

config_file = "config.pkl"
verbose = True

config = load_config(config_file)

image_types = config["image_types"]
save_csv = config["save_csv"]
write_ids = config["write_ids"]
mode = config["mode"]
raw_data = config["raw_data"]
temp_data = config["temp_data"]
clean_data = config["clean_data"]
detection_data = config["detection_data"]
results_data = config["results_data"]
default_model = config["default_model"]
default_base_model = config["default_base_model"]
default_yaml = config["default_yaml"]
max_epochs = config["max_epochs"]
models_path = config["models_path"]
patience = config["patience"]
save_period = config["save_period"]
resize_stat_name = config["resize_stat_name"]
base_class = config["base_class"]


def run_detections(models_path, model_name, confidence, save_results, detection_path, results_data, save_csv=False, write_ids=False, retain_output=False):
    print("Running detection...")
    run_dir, results = run_model(models_path, model_name, confidence, save_results, detection_path, detection_path, detection_data)
    
    if not retain_output:
        try:
            shutil.rmtree(run_dir)
        except:
            pass

    if save_csv:
        save_obbs_to_csv(results, detection_path, results_data, write_ids)

def train_detection(model_name, base_model, data_file, max_epochs, models_path, patience=100, save_period=10):
    print("Training detection...")
    train_model(model_name, base_model, data_file, max_epochs, patience=patience, save_period=save_period, models_path=models_path, degrees=180, copy_paste=0.5, mixup=0.5, flipud=0.5, multi_scale=True)

print("Pre-processing images...")
pre_process_images(raw_data, temp_data, image_metadata_filename)
print("Generating labels...")
generate_labels(temp_data, clean_data, oriented_bb=True)
try:
    shutil.rmtree(temp_data)
except:
    pass

if mode == "detect":
    run_detections("models", default_model, 0.5, True, clean_data, results_data, save_csv=save_csv, write_ids=write_ids)
elif mode == "train":
    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_name = "yolo11n_particles-obb" + current_time + ".pt"
    train_detection(model_name, default_base_model, default_yaml, max_epochs, models_path, patience=patience, save_period=save_period)

write_config(config, config_file)