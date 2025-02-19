import os
import numpy as np
from math import ceil, floor
import cv2
from image_management import get_image_path

image_types = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
default_image_type = ".png"
images_dirs = ["clean_data-obb"]
labels_dirs = ["clean_data-obb"]
output_images_dirs = ["clean_data-obb_resized"]
output_labels_dirs = ["clean_data-obb_resized"]

def read_results(dir_path, verbose=False):
    mixed_labels_exception = "Error: Mixed OBB labels and non-OBB labels."

    obb_labels = None

    results = {}
    for file in os.listdir(dir_path):
        result = []
        if file.endswith(".txt"):
            with open(os.path.join(dir_path, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    line = [float(x) for x in line]
                    if len(line) == 9:
                        obb_labels = True
                        line = line[1:]
                        result.append(line)
                    elif len(line) == 5 and obb_labels == True:
                        raise Exception(mixed_labels_exception)
                    if len(line) == 5:
                        obb_labels = False
                        result.append(line)
                    elif len(line) == 9 and obb_labels == False:
                        raise Exception(mixed_labels_exception)
        file = os.path.splitext(file)[0]
        results[file] = result
    
    if verbose:
        print("obb_labels:", obb_labels)
        print("results:", results)

    return results, obb_labels

def get_dims_obb(label):
    x1, y1, x2, y2, x3, y3, x4, y4 = label
    width = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    height = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    return width, height

def get_dims_obbs(labels):
    dims = []
    for label in labels:
        width, height = get_dims_obb(label)
        dims.append([width, height])
    return dims

def get_size_obbs(labels):
    dims = get_dims_obbs(labels)
    sizes = []
    for dim in dims:
        size = dim[0] * dim[1]
        sizes.append(size)
    return sizes

# get mean, median, standard deviation, variance, min, max, and count of results for each result and for all results
def get_results_stats(results, obb_labels=True, verbose=False):
    individual_stats = {}

    all_sizes = []

    for key in results:
        result = results[key]

        if obb_labels:
            result_sizes = get_size_obbs(result)
        else:
            raise Exception("Error: Not implemented.")

        all_sizes.extend(result_sizes)

        result_sizes = np.array(result_sizes)

        # sqrt the sizes to get the average size of the particles
        result_sizes = np.sqrt(result_sizes)

        # get stats
        mean = np.mean(result_sizes, axis=0)
        median = np.median(result_sizes, axis=0)
        std = np.std(result_sizes, axis=0)
        var = np.var(result_sizes, axis=0)
        min_ = np.min(result_sizes, axis=0)
        max_ = np.max(result_sizes, axis=0)
        count = len(result_sizes)

        individual_stats[key] = {"mean": mean, "median": median, "std": std, "var": var, "min": min_, "max": max_, "count": count}

    np.array(all_sizes)

    # sqrt the sizes to get the average size of the particles
    all_sizes = np.sqrt(all_sizes)

    # get stats
    all_sizes = np.array(all_sizes)
    mean = np.mean(all_sizes, axis=0)
    median = np.median(all_sizes, axis=0)
    std = np.std(all_sizes, axis=0)
    var = np.var(all_sizes, axis=0)
    min_ = np.min(all_sizes, axis=0)
    max_ = np.max(all_sizes, axis=0)
    count = len(all_sizes)
    general_stats = {"mean": mean, "median": median, "std": std, "var": var, "min": min_, "max": max_, "count": count}

    if verbose:
        print("individual_stats:", individual_stats)
        print("general_stats:", general_stats)

    return individual_stats, general_stats

# resizes the image to be bigger by adding white space
def fill_image(image, resize):
    if resize == 1:
        return image
    if resize < 1:
        raise Exception("Error: Resize must be bigger than one.")
    height, width, channels = image.shape
    new_height = int(height * resize)
    new_width = int(width * resize)
    new_image = np.ones((new_height, new_width, channels), np.uint8) * 255
    new_image[0:height, 0:width] = image
    return new_image

def fill_label(label, resize):
    if resize == 1:
        return label
    if resize < 1:
        raise Exception("Error: Resize must be bigger than one.")
    x1, y1, x2, y2, x3, y3, x4, y4 = label
    new_x1 = x1 * resize
    new_y1 = y1 * resize
    new_x2 = x2 * resize
    new_y2 = y2 * resize
    new_x3 = x3 * resize
    new_y3 = y3 * resize
    new_x4 = x4 * resize
    new_y4 = y4 * resize
    return [new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4]

def fill_labels(labels, resize):
    new_labels = []
    for label in labels:
        new_label = fill_label(label, resize)
        new_labels.append(new_label)
    return new_labels

def break_image(image, labels, resize):
    if resize == 1:
        return [image]
    if resize >= 1:
        raise Exception("Error: Resize must be smaller than one.")
    height, width, channels = image.shape
    new_height = int(height * resize)
    new_width = int(width * resize)
    width_resize = width / new_width
    height_resize = height / new_height
    images = []
    new_labels = []
    for i in range(ceil(height/new_height)):
        for j in range(ceil(width/new_width)):
            new_image = np.ones((new_height, new_width, channels), np.uint8) * 255
            starting_height = i * new_height
            starting_width = j * new_width
            next_height = min(starting_height + new_height, height)
            next_width = min(starting_width + new_width, width)
            if starting_height >= height or starting_width >= width:
                continue
            slice_height = next_height - starting_height
            slice_width = next_width - starting_width
            new_image[0:slice_height, 0:slice_width] = image[starting_height:next_height, starting_width:next_width]
            images.append(new_image)

            image_labels = []
            for label in labels:
                normalized_starting_width = starting_width / width
                normalized_starting_height = starting_height / height
                normalized_next_width = next_width / width
                normalized_next_height = next_height / height
                x1, y1, x2, y2, x3, y3, x4, y4 = label
                if not (x1 >= normalized_starting_width and x1 < normalized_next_width and y1 >= normalized_starting_height and y1 < normalized_next_height):
                    continue
                if not (x2 >= normalized_starting_width and x2 < normalized_next_width and y2 >= normalized_starting_height and y2 < normalized_next_height):
                    continue
                if not (x3 >= normalized_starting_width and x3 < normalized_next_width and y3 >= normalized_starting_height and y3 < normalized_next_height):
                    continue
                if not (x4 >= normalized_starting_width and x4 < normalized_next_width and y4 >= normalized_starting_height and y4 < normalized_next_height):
                    continue
                new_x1 = (x1 - normalized_starting_width) / width_resize
                new_y1 = (y1 - normalized_starting_height) / height_resize
                new_x2 = (x2 - normalized_starting_width) / width_resize
                new_y2 = (y2 - normalized_starting_height) / height_resize
                new_x3 = (x3 - normalized_starting_width) / width_resize
                new_y3 = (y3 - normalized_starting_height) / height_resize
                new_x4 = (x4 - normalized_starting_width) / width_resize
                new_y4 = (y4 - normalized_starting_height) / height_resize
                new_label = [new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4]
                image_labels.append(new_label)
                
            new_labels.append(image_labels)
    
    return images, new_labels

def break_label(label, resize):
    if resize == 1:
        return [label]
    if resize >= 1:
        raise Exception("Error: Resize must be smaller than one.")
    x1, y1, x2, y2, x3, y3, x4, y4 = label

    new_x1 = (x1 - floor(x1/resize) * resize) / resize
    new_y1 = (y1 - floor(y1/resize) * resize) / resize
    new_x2 = (x2 - floor(x2/resize) * resize) / resize
    new_y2 = (y2 - floor(y2/resize) * resize) / resize
    new_x3 = (x3 - floor(x3/resize) * resize) / resize
    new_y3 = (y3 - floor(y3/resize) * resize) / resize
    new_x4 = (x4 - floor(x4/resize) * resize) / resize
    new_y4 = (y4 - floor(y4/resize) * resize) / resize

    return [[new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4]]
    
def resize_image_and_labels(image, labels, resize):
    if resize == 1:
        return [image], labels
    if resize >= 1:
        new_image = fill_image(image, resize)
        new_labels = fill_labels(labels, resize)
        return [new_image], [new_labels]
    if resize < 1:
        new_images, new_labels = break_image(image, labels, resize)
        return new_images, new_labels

def resize_images(images_dirs, labels_dirs, output_images_dirs, output_labels_dirs, resize_by="median", base_class=0, acceptable_error=0.001):
    for images_dir, labels_dir, output_images_dir, output_labels_dir in zip(images_dirs, labels_dirs, output_images_dirs, output_labels_dirs):
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)
        if not os.path.exists(output_labels_dir):
            os.makedirs(output_labels_dir)
        if not os.path.exists(images_dir):
            raise Exception("Error: Images directory not found.")
        if not os.path.exists(labels_dir):
            raise Exception("Error: Labels directory not found.")

        images_dir_path = images_dir

        labels_dir_path = labels_dir
        results, obb_labels = read_results(labels_dir_path, verbose=False)
        individual_stats, general_stats = get_results_stats(results, obb_labels=obb_labels, verbose=False)

        resize_operations = {}

        for key in individual_stats:
            individual_stat = individual_stats[key]
            individual_resize_stat = individual_stat[resize_by]
            general_stats_resize_stat = general_stats[resize_by]
            resize_operations[key] = individual_resize_stat/general_stats_resize_stat

        # if bigger than one, make image smaller, if smaller than one, fill image to be able to break it down into smaller images (int)
        for key in resize_operations:
            resize_factor = resize_operations[key]

            output_image_filename = key + default_image_type
            output_image_path = os.path.join(output_images_dir, output_image_filename)

            image_path = get_image_path(images_dir_path, key)

            if image_path == None:
                print("key:", key)
                raise Exception("Error: Image not found.")

            image = cv2.imread(image_path)

            if abs(resize_factor - 1) < acceptable_error:
                cv2.imwrite(output_image_path, image)
            else:
                print("key:", key)
                print("resize_factor:", resize_factor)
                resized_images, resized_labels = resize_image_and_labels(image, results[key], resize_factor)
                counter = 0
                for resized_image, resized_image_labels in zip(resized_images, resized_labels):
                    resized_image_filename = key + "_" + str(counter) + default_image_type
                    resized_image_path = os.path.join(output_images_dir, resized_image_filename)
                    cv2.imwrite(resized_image_path, resized_image)
                    resized_image_labels_filename = key + "_" + str(counter) + ".txt"
                    with open(os.path.join(output_labels_dir, resized_image_labels_filename), "w") as f:
                        for label in resized_image_labels:
                            f.write(str(base_class) + " " + " ".join([str(x) for x in label]) + "\n")
                    counter += 1

if __name__ == "__main__":
    resize_images(images_dirs, labels_dirs, output_images_dirs, output_labels_dirs, resize_by="median")