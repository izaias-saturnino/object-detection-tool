import os
import time
import pickle

def load_config(config_file, verbose=False):
    config = {}

    image_types = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    
    save_csv = True
    write_ids = True

    mode = "detect" # "detect" or "train"

    raw_data = "raw_data"
    temp_data = "temp_data"
    clean_data = "clean_data"

    default_model = "yolo11n-obb_test.pt_20250218-072546.pt"
    default_base_model = "yolo11n-obb.pt"

    default_yaml = "data_TEM.yaml"
    max_epochs = 5000
    models_path = "models"
    patience = 100
    save_period = 10

    resize_stat_name = "median"
    
    # read config file with pickle
    if os.path.exists(config_file):
        with open(config_file, "rb") as f:
            config = pickle.load(f)
    else:
        if verbose:
            print("Config file not found.")

    if "image_types" not in config:
        config["image_types"] = image_types
    if "save_csv" not in config:
        config["save_csv"] = save_csv
    if "write_ids" not in config:
        config["write_ids"] = write_ids
    if "mode" not in config:
        config["mode"] = mode
    if "raw_data" not in config:
        config["raw_data"] = raw_data
    if "temp_data" not in config:
        config["temp_data"] = temp_data
    if "clean_data" not in config:
        config["clean_data"] = clean_data
    if "default_model" not in config:
        config["default_model"] = default_model
    if "default_base_model" not in config:
        config["default_base_model"] = default_base_model
    if "default_yaml" not in config:
        config["default_yaml"] = default_yaml
    if "max_epochs" not in config:
        config["max_epochs"] = max_epochs
    if "models_path" not in config:
        config["models_path"] = models_path
    if "patience" not in config:
        config["patience"] = patience
    if "save_period" not in config:
        config["save_period"] = save_period
    if "resize_stat_name" not in config:
        config["resize_stat_name"] = resize_stat_name

    return config

def write_config(config, config_file):
    with open(config_file, "wb") as f:
        pickle.dump(config, f)