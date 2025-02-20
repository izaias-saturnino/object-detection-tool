import os
import cv2
import math
import numpy as np
from image_management import find_images, get_image_path
import shutil

image_types = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
default_image_type = ".png"
base_class = 0

def cut_img(img):
    # Find the first row where all pixels are black
    white_row = np.all(img[:, :, 0] == 0, axis=1)
    cut = np.argmax(white_row) if np.any(white_row) else None

    # Remove this line and all pixels below it
    new_img = img[:cut, :, :] if cut is not None else img
    
    return new_img

def image_gradient(image, ksize=3):
    # use sobel to calculate the gradient
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    grad_sum = sobelx**2 + sobely**2
    
    grad = np.sqrt(grad_sum)

    return grad

def border_mask(image, weight=1, invert=False):
    grad = image_gradient(image)

    grad = grad / np.max(grad)

    # convert to one channel if it is three

    grayscale = False
    if len(grad.shape) == 3:
        grayscale = True
        grad = np.mean(grad, axis=2)
    
    # if the gradient is below gray, invert it
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            if grad[i, j] < 0.5:
                grad[i, j] = 1 - grad[i, j]
    
    # normalize the gradient in the float range 0-1
    grad = grad - 0.5
    grad = grad / 0.5

    # normalize the gradient in the range 0-255
    grad = grad * 255 * weight

    if grayscale:
        # convert back to three channels
        grad = np.stack([grad, grad, grad], axis=2)

    if invert:
        # invert the image
        grad = 255 - grad

    return grad

# not used
def clean_img(image, weight=1, invert=True):

    # get the border mask
    border_mask_img = border_mask(image, weight=weight, invert=invert)

    # reverse the mask
    reverse_mask = 255 - border_mask_img

    # convert to float from 0 to 1 using cv2
    border_mask_img = border_mask_img / 255.0
    reverse_mask = reverse_mask / 255.0

    # blur the image
    blurred = cv2.GaussianBlur(image, (5,5), 0)

    reverse_blurred = blurred * reverse_mask

    borders = image * border_mask_img

    final_img = cv2.addWeighted(borders, 1, reverse_blurred, 1, 0)
    final_img = final_img.astype(np.uint8)

    return final_img

def pre_process_image(origin_image_path, destination_path=None, image_types=image_types):

    image = os.path.basename(origin_image_path)

    image_name, image_extension = os.path.splitext(image)

    if image_extension not in image_types:
        return

    if destination_path is not None:
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

    # read the image without red lines
    image_path = origin_image_path
    img = cv2.imread(image_path)

    img = cut_img(img)

    # save the image
    if destination_path is not None:
        new_image_filename = image_name + default_image_type
        new_image_path = os.path.join(destination_path, new_image_filename)
        cv2.imwrite(new_image_path, img)

    return img

def pre_process_images(input_path, output_path, image_metadata_filename, verbose=False, image_types=image_types):
    print("Finding new images...")
    new_images = find_images(input_path, image_metadata_filename=image_metadata_filename, verbose=verbose, image_types=image_types)
    if len(new_images) > 0:
        if verbose:
            print("New images found")
            print("Pre-process input path:", input_path)
            print("Pre-process output path:", output_path)
        for image in new_images:
            pre_process_image(os.path.join(input_path, image), output_path)
    else:
        if verbose:
            print("No new images found.")

def generate_labels(origin_path, destination_path, oriented_bb=True, diameter_annotation=True, draw_obj_bboxes=False, image_types=image_types, verbose=False):
    if os.path.exists(origin_path) == False:
        print("Origin path does not exist.")
        return
    if os.path.exists(destination_path) == False:
        os.makedirs(destination_path)

    # the folder contains images with red lines and the same images without red lines
    # the red lines are the diameter of the object
    # the goal is to extract the position and size of the diameters from the images and create the labels for the images with red lines

    # the images that end with xxxx are the images with red lines

    # read the images from the folder
    images = os.listdir(origin_path)
    images = [i for i in images if os.path.splitext(i)[1] in image_types]
    images = [i for i in images if os.path.splitext(i)[0].lower().endswith("xxxx")]

    # create the labels for the images
    for image in images:

        bboxes = []

        image_name = os.path.splitext(image)[0]
        image_extension = os.path.splitext(image)[1]

        # check if the image with red lines has a corresponding image without red lines
        image_no_lines = image_name[:-4]

        image_no_lines_path = get_image_path(origin_path, image_no_lines, image_types=image_types)
        
        if verbose:
            print(image_no_lines_path)

        if image_no_lines_path is None:
            continue

        image_no_lines_filename = os.path.basename(image_no_lines_path)

        # read the image without red lines
        img_no_lines = cv2.imread(image_no_lines_path)

        new_img_no_lines = img_no_lines

        # save the image
        cv2.imwrite(os.path.join(destination_path, image_no_lines_filename), new_img_no_lines)

        # read the image
        image_path = os.path.join(origin_path, image)

        img = cv2.imread(image_path)

        # this line is needed because cv2 reads the image in BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # extract the red lines from the image
        img_red_lines = cv2.inRange(img, (200, 0, 0), (255, 50, 50))

        # create contours from the red lines
        contours, _ = cv2.findContours(img_red_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x = x + w / 2
            y = y + h / 2

            if diameter_annotation:
                d = math.sqrt(w**2 + h**2)
                w = d
                h = d

            # normalize the values
            x = x / new_img_no_lines.shape[1]
            y = y / new_img_no_lines.shape[0]
            w = w / new_img_no_lines.shape[1]
            h = h / new_img_no_lines.shape[0]

            bboxes.append((x, y, w, h))

        # calculate the average size of the bounding boxes

        average_size = 0
        for bbox in bboxes:
            x, y, w, h = bbox
            average_size += w
        try:
            average_size /= len(bboxes)
        except ZeroDivisionError:
            average_size = 0
        average_size = int(average_size * new_img_no_lines.shape[1])

        new_file_name = image_name[:-4] + ".txt"

        # save the labels to a file
        labels_path = os.path.join(destination_path, new_file_name)
        with open(labels_path, "w") as file:
            for bbox in bboxes:
                x, y, w, h = bbox
                if oriented_bb:
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y - h / 2
                    x3 = x + w / 2
                    y3 = y + h / 2
                    x4 = x - w / 2
                    y4 = y + h / 2
                    file.write(f"{base_class} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
                else:
                    file.write(f"{base_class} {x} {y} {w} {h}\n")

        # draw the bounding boxes on the image and save it
        if draw_obj_bboxes:
            for bbox in bboxes:
                x, y, w, h = bbox
                x = int(x * new_img_no_lines.shape[1])
                y = int(y * new_img_no_lines.shape[0])
                w = int(w * new_img_no_lines.shape[1])
                h = int(h * new_img_no_lines.shape[0])
                cv2.rectangle(new_img_no_lines, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)

        drawn_image_name = image_name[:-4] + "_drawn" + image_extension
        if draw_obj_bboxes:
            cv2.imwrite(os.path.join(destination_path, drawn_image_name), new_img_no_lines)

if __name__ == "__main__":
    print("Pre-processing images...")
    pre_process_images("raw_data", "temp_data", "image_metadata.pkl")
    print("Generating labels...")
    generate_labels("temp_data", "clean_data", oriented_bb=True)
    try:
        shutil.rmtree("temp_data")
    except:
        pass