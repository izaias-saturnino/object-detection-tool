import os
import cv2
import numpy as np
from math import ceil, floor

def to_el_list(obb):
    if len(obb) == 8:
        return obb
    elif len(obb) == 4:
        new_obb = []
        for el in obb:
            new_obb.append(el[0])
            new_obb.append(el[1])
        return new_obb
    else:
        raise ValueError("Invalid number of elements in the bounding box.")

def rotated_rect_area(obb):
    return obb.size[0] * obb.size[1]

def obb_intersection(obb1, obb2):
    # Get the four vertices of the rotated rectangles
    vertices1 = cv2.boxPoints(obb1)
    vertices2 = cv2.boxPoints(obb2)
    
    # Find the intersection polygon
    intersection_polygon = cv2.intersectConvexConvex(vertices1, vertices2)
    
    if intersection_polygon[0] > 0:
        # Calculate the area of the intersection polygon
        intersection_area = cv2.contourArea(intersection_polygon[1])
    else:
        intersection_area = 0.0
    
    return intersection_area

def get_IoU_oriented(box1, box2):

    box1 = to_el_list(box1)
    box2 = to_el_list(box2)

    x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1 = box1
    x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1 = int(x1_1), int(y1_1), int(x2_1), int(y2_1), int(x3_1), int(y3_1), int(x4_1), int(y4_1)

    x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2 = box2
    x1_2, y1_2, x2_2, y2_2, x3_2, y3_2, x4_2, y4_2 = int(x1_2), int(y1_2), int(x2_2), int(y2_2), int(x3_2), int(y3_2), int(x4_2), int(y4_2)

    center1 = ((x1_1 + x3_1) / 2, (y1_1 + y3_1) / 2)
    center2 = ((x1_2 + x3_2) / 2, (y1_2 + y3_2) / 2)
    size1 = ((x1_1 - x2_1) ** 2 + (y1_1 - y2_1) ** 2) ** 0.5, ((x1_1 - x4_1) ** 2 + (y1_1 - y4_1) ** 2) ** 0.5
    size2 = ((x1_2 - x2_2) ** 2 + (y1_2 - y2_2) ** 2) ** 0.5, ((x1_2 - x4_2) ** 2 + (y1_2 - y4_2) ** 2) ** 0.5
    angle1 = np.arctan2(y2_1 - y1_1, x2_1 - x1_1) * 180 / np.pi
    angle2 = np.arctan2(y2_2 - y1_2, x2_2 - x1_2) * 180 / np.pi

    obb1 = cv2.RotatedRect(center1, size1, angle1)
    obb2 = cv2.RotatedRect(center2, size2, angle2)

    cv2.rotatedRectangleIntersection(obb1, obb2)

    obb1_area = rotated_rect_area(obb1)
    obb2_area = rotated_rect_area(obb2)
    intersection_area = obb_intersection(obb1, obb2)

    union_area = obb1_area + obb2_area - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area

def get_max_dims_obb(obb):
    x1 = obb[0][0]
    y1 = obb[0][1]
    x2 = obb[1][0]
    y2 = obb[1][1]
    x3 = obb[2][0]
    y3 = obb[2][1]
    x4 = obb[3][0]
    y4 = obb[3][1]
    
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    width = x_max - x_min
    height = y_max - y_min
    return width, height

def obb_grid(obbs, grid_width, grid_height):
    max_obj_size = 0
    centroids = []
    for obb in obbs:
        x1 = obb[0][0]
        y1 = obb[0][1]
        x2 = obb[1][0]
        y2 = obb[1][1]
        x3 = obb[2][0]
        y3 = obb[2][1]
        x4 = obb[3][0]
        y4 = obb[3][1]
        
        centroid = (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4
        centroids.append(centroid)

        width, height = get_max_dims_obb(obb)
        max_obj_size = max(max_obj_size, width, height)

    if max_obj_size == 0:
        return [[[]]]

    cell_size = max_obj_size

    cell_width = ceil(grid_width / cell_size)
    cell_height = ceil(grid_height / cell_size)

    grid = [[[] for _ in range(cell_width)] for _ in range(cell_height)]

    for i, obb in enumerate(obbs):
        centroid = centroids[i]
        x, y = centroid
        cell_x = floor(x / cell_size)
        cell_y = floor(y / cell_size)
        try:
            grid[cell_y][cell_x].append(obb)
        except Exception as e:
            print("cell_x:", cell_x)
            print("cell_y:", cell_y)
            print("grid_width:", grid_width)
            print("grid_height:", grid_height)
            print("cell_width:", cell_width)
            print("cell_height:", cell_height)
            raise e

    return grid

def obbs_NMS(obbs, image_width, image_height, iou_threshold=0.5):
    new_obbs = []

    grid = obb_grid(obbs, image_width, image_height)

    grid_cell_width = len(grid[0])
    grid_cell_height = len(grid)

    for y in range(grid_cell_height):
        for x in range(grid_cell_width):
            obb1_index = 0
            while obb1_index < len(grid[y][x]):
                obb1 = grid[y][x][obb1_index]

                deleted = False
                has_intersection = False

                for y_offset in range(-1, 2):
                    for x_offset in range(-1, 2):
                        new_y = y + y_offset
                        new_x = x + x_offset
                        if new_y >= 0 and new_y < grid_cell_height and new_x >= 0 and new_x < grid_cell_width:
                            for obb2_index in range(len(grid[new_y][new_x])):
                                obb2 = grid[new_y][new_x][obb2_index]

                                if obb1_index == obb2_index and obb1 == obb2:
                                    continue

                                IoU = get_IoU_oriented(obb1, obb2)
                                if IoU > iou_threshold:
                                    del grid[new_y][new_x][obb1_index]
                                    deleted = True
                                    
                                    has_intersection = True
                                    break
                            if has_intersection:
                                break
                    if has_intersection:
                        break

                if not has_intersection:
                    new_obbs.append(obb1)

                    if not deleted:
                        del grid[y][x][obb1_index]
                        deleted = True

                if deleted:
                    obb1_index -= 1
                
                obb1_index += 1

    return new_obbs

def restore_broken_detection(results, break_metadatas):
    new_results = {}

    broken_keys = []

    for key in break_metadatas:
        break_metadata = break_metadatas[key]
        height = break_metadata["height"]
        width = break_metadata["width"]
        height_step = break_metadata["height_step"]
        width_step = break_metadata["width_step"]
        block_height = break_metadata["block_height"]
        block_width = break_metadata["block_width"]

        original_obbs = []

        for i in range(0, block_height):
            for j in range(0, block_width):
                image_index = i * block_width + j + 1
                image_name = f"{key}_{image_index}"
                
                for result in results:
                    result_image_name = os.path.basename(result.path)
                    result_image_name, _ = os.path.splitext(result_image_name)
                    if result_image_name == image_name:
                        broken_keys.append(image_name)
                        obbs = result.obb.xyxyxyxy
                        
                        for obb in obbs:
                            x1 = obb[0][0]
                            y1 = obb[0][1]
                            x2 = obb[1][0]
                            y2 = obb[1][1]
                            x3 = obb[2][0]
                            y3 = obb[2][1]
                            x4 = obb[3][0]
                            y4 = obb[3][1]

                            x1 = x1 + j * width_step
                            y1 = y1 + i * height_step
                            x2 = x2 + j * width_step
                            y2 = y2 + i * height_step
                            x3 = x3 + j * width_step
                            y3 = y3 + i * height_step
                            x4 = x4 + j * width_step
                            y4 = y4 + i * height_step
                            obb = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                            original_obbs.append(obb)
                        break

        new_obbs = obbs_NMS(original_obbs, width, height)

        new_results[key] = new_obbs

    for result in results:
        path = result.path
        image_name = os.path.basename(path)
        image_name, _ = os.path.splitext(image_name)
        if image_name not in broken_keys:
            new_results[image_name] = result.obb.xyxyxyxy

    return new_results

def save_obbs_to_csv(results, input_path, output_path, write_ids=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(input_path):
        raise Exception("Input path does not exist.")

    for key in results:
        result = results[key]
        image_name = key
        image_path = os.path.join(input_path, image_name + ".png")
        csv_path = os.path.join(output_path, image_name + ".csv")

        image = None

        if write_ids:
            image = cv2.imread(image_path)
            image_filename = os.path.basename(image_path)
            image_name, image_extension = os.path.splitext(image_filename)
            image_filename = image_name + "_with_obbs" + image_extension
            image_output_path = os.path.join(output_path, image_filename)

        image_name = os.path.basename(image_path)
        obbs = result

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