import os
import pickle

image_types = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def get_image_path(images_dir_path, image_name, image_types=image_types):
    for image_type in image_types:
        image_path = os.path.join(images_dir_path, image_name + image_type)
        if os.path.exists(image_path):
            return image_path
    return None

def load_image_metadata(path, image_metadata_filename):
    if image_metadata_filename is None:
        return []
    if os.path.exists(path):
        image_metadata = pickle.load(open(os.path.join(path, image_metadata_filename), "rb"))
        return image_metadata
    else:
        return []

def save_image_metadata(path, images, image_metadata_filename):
    pickle.dump(images, open(os.path.join(path, image_metadata_filename), "wb"))

def find_images(path, image_metadata_filename, verbose=False, image_types=image_types):
    if not os.path.exists(path):
        return []
    images = []
    files = os.listdir(path)
    image_metadata = []

    if verbose:
        print("Files:", files)
        print("Image types:", image_types)
        print("path:", path)
        print("image_metadata_filename:", image_metadata_filename)

    if image_metadata_filename is None or image_metadata_filename in files:
        image_metadata = load_image_metadata(path, image_metadata_filename)
        if verbose:
            print("Image metadata file found")
    
    # remove non-image files
    for file in files:
        _, file_extension = os.path.splitext(file)
        if file_extension in image_types:
            images.append(file)

    new_images = []
            
    if image_metadata is not None:
        for image in images:
            if image not in image_metadata:
                new_images.append(image)
    else:
        new_images = images

    if len(new_images) > 0:
        if image_metadata_filename is not None:
            save_image_metadata(path, images, image_metadata_filename)

    return new_images