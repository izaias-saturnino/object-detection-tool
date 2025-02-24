import os
import shutil
import numpy as np
from ultralytics import YOLO
import time
from math import ceil
from sklearn.linear_model import LinearRegression
from ultralytics import settings
import pickle

def read_model_metadata(model_path, model):
    model_name, _ = os.path.splitext(model)
    metadata_filename = model_name + ".pkl"
    metadata_path = os.path.join(model_path, metadata_filename)
    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)
    return metadata

def test_models(models_path, models, save_results, save_csv=True, write_ids=True, split="test"):
    run_dirs = []
    for model in models:
        model_path = os.path.join(models_path, model)
        model = YOLO(model_path)
        results = model.val(data="data_TEM.yaml", save=save_results, augment=True, show_labels=False, mode="val", verbose=False)

        loss = results.box.ap50[0]
        print("Model:", model)
        print("Loss:", loss)

        current_time = time.strftime("%Y%m%d-%H%M%S")

        try:
            run_dir = "runs_" + model + "_" + current_time
            shutil.move("runs", run_dir)
            run_dirs.append(run_dir)
        except:
            pass

        if save_csv:
            pass
            #for result in results:
            #    pass # TODO: save results to csv alongside the image names (write ids if write_ids is True)

    return run_dirs

def train_model(model_name, base_model, data_file, max_epochs, patience=100, save_period=10, models_path="models", degrees=180, copy_paste=0.5, mixup=0.5, flipud=0.5, multi_scale=True):
    model = YOLO(base_model)
    
    model.train(data=data_file, epochs=max_epochs, patience=patience, verbose=False, show_labels=False, degrees=degrees, copy_paste=copy_paste, mixup=mixup, flipud=flipud, multi_scale=multi_scale, save_period=save_period)

    if os.path.exists(models_path) == False:
        os.makedirs(models_path)
    
    current_time = time.strftime("%Y%m%d-%H%M%S")

    model.save(models_path + "/" + model_name + "_" + current_time + ".pt")

    return model

if __name__ == "__main__":
    # train_model("yolo11n-obb_test.pt", "yolo11n-obb.pt", "data_TEM.yaml", 600)
    test_models("models", ["yolov8n_black_particles-obb_phase_1.pt", "yolo11n-obb_test.pt_20250218-072546.pt"], save_results=True, split="test")
    test_models("models", ["yolov8n_black_particles-obb_phase_1.pt", "yolo11n-obb_test.pt_20250218-072546.pt"], save_results=True, split="val")