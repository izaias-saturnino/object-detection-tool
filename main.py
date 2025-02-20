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

image_metadata_filename = "image_metadata.pkl"
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
default_model = config["default_model"]
default_base_model = config["default_base_model"]
default_yaml = config["default_yaml"]
max_epochs = config["max_epochs"]
models_path = config["models_path"]
patience = config["patience"]
save_period = config["save_period"]
resize_stat_name = config["resize_stat_name"]
base_class = config["base_class"]

def save_obbs_to_csv(results, output_path, write_ids=False):
    for i, result in enumerate(results):
        image_path = result.path
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        csv_path = os.path.join(output_path, image_name + ".csv")

        image = None

        if write_ids:
            image = cv2.imread(image_path)
            image_filename = os.path.basename(image_path)
            image_name, image_extension = os.path.splitext(image_filename)
            image_filename = image_name + "_with_obbs" + image_extension
            image_output_path = os.path.join(output_path, image_filename)

        image_name = os.path.basename(image_path)
        obbs = results[i].obb.xyxyxyxy

        # order the obbs by their y1 value and then by their x1 value
        obbs = sorted(obbs, key=lambda obb: (obb[0][1], obb[0][0]))

        with open(csv_path, "w") as file:
            file.write("obb_id, height, width, angle1, angle2, x1, y1, x2, y2, x3, y3, x4, y4\n")
            obb_id = 1
            for obb in obbs:
                x1 = obb[0][0]
                y1 = obb[0][1]
                x2 = obb[1][0]
                y2 = obb[1][1]
                x3 = obb[2][0]
                y3 = obb[2][1]
                x4 = obb[3][0]
                y4 = obb[3][1]
                # calculate the height and width of the obb in pixels
                height = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
                width = ((y3 - y2)**2 + (x3 - x2)**2)**0.5
                # calculate the angle of the obb based on the first two points
                angle1 = np.arctan2(y2 - y1, x2 - x1)
                angle2 = np.arctan2(y3 - y2, x3 - x2)
                file.write(f"{obb_id}, {height}, {width}, {angle1}, {angle2}, {x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}\n")
            
                if write_ids:
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    x3 = int(x3)
                    y3 = int(y3)
                    x4 = int(x4)
                    y4 = int(y4)
                    # draw the bounding boxes on the image
                    height, width, _ = image.shape
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.line(image, (x2, y2), (x3, y3), (0, 255, 0), 2)
                    cv2.line(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.line(image, (x4, y4), (x1, y1), (0, 255, 0), 2)

                    # draw a number in the middle of the bounding box
                    x = int((x1 + x2 + x3 + x4) / 4)
                    y = int((y1 + y2 + y3 + y4) / 4)
                    cv2.putText(image, str(obb_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                obb_id += 1

            if write_ids:
                # save the image with the bounding boxes
                cv2.imwrite(image_output_path, image)

def retain_output(run_dir):
    pass # TODO: retain output

def run_detections(models_path, model_name, confidence, save_results, detection_path, save_csv=False, write_ids=False):
    print("Running detection...")
    run_dir, results = run_model(models_path, model_name, confidence, save_results, detection_path, detection_path, detection_data)
    retain_output(run_dir)

    if save_csv:
        save_obbs_to_csv(results, detection_data, write_ids)

def train_detection(model_name, base_model, data_file, max_epochs, models_path, patience=100, save_period=10):
    print("Training detection...")
    train_model(model_name, base_model, data_file, max_epochs, patience=patience, save_period=save_period, models_path=models_path, degrees=180, copy_paste=0.5, mixup=0.5, flipud=0.5, multi_scale=True)

print("Pre-processing images...")
pre_process_images(raw_data, temp_data, "image_metadata.pkl")
print("Generating labels...")
generate_labels(temp_data, clean_data, oriented_bb=True)
try:
    shutil.rmtree(temp_data)
except:
    pass

if mode == "detect":
    run_detections("models", default_model, 0.5, True, clean_data, save_csv=save_csv, write_ids=write_ids)
elif mode == "train":
    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_name = "yolo11n_particles-obb" + current_time + ".pt"
    train_detection(model_name, default_base_model, default_yaml, max_epochs, models_path, patience=patience, save_period=save_period)

write_config(config, config_file)