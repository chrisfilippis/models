from read_roi import read_roi_zip, read_roi_file
from os import listdir
import os
from os.path import isfile, join
import json
from shutil import copyfile
import zipfile
import cv2
import numpy as np


def get_input_files(directory_path, extension='zip'):
    return [f for f in listdir(directory_path) if isfile(join(directory_path, f)) and f.endswith('.' + extension)]


def get_polygons(roi_file):
    return [get_polygon_info(polygon) for polygon in roi_file]


def get_polygon_info(polygon):
    name = polygon[1]['name']
    x = polygon[1]['x']
    y = polygon[1]['y']
    return name, x, y


def get_roi_files_from_zipfile(annotation_file_path, filter_clause='superpixel'):
    annotation_polygon = filter_zip(annotation_file_path)
    roi = list(annotation_polygon.items())
    polygons = get_polygons(roi)
    return [poly for poly in polygons if filter_clause not in poly[0]]


def filter_zip(zip_path):
    from collections import OrderedDict
    rois = OrderedDict()
    zf = zipfile.ZipFile(zip_path)
    for n in zf.namelist():
        if n.endswith('.roi'):
            rois.update(read_roi_file(zf.open(n)))
    return rois


def process_regions_of_interest(roi_files):
    regions = []

    for region in roi_files:
        dental_class = int(region[0].split('-')[0])

        if dental_class == 0:
            continue

        if dental_class > 6:
            dental_class = 6

        regions.append((dental_class, region[1], region[2]))

    return regions


def create_image_json_data_for_image(image_name, name_mapping_dic):
    return {
        "id": name_mapping_dic[image_name],
        "width": 1024,
        "height": 768,
        "file_name": image_name,
        "date_captured": "2013-11-15 02:41:42"
    }


def create_annotation_json_data_for_image(image_name, regions, name_mapping_dic):
    regions_data = []
    i = 0
    for region in regions:
        annot = []

        for ii in range(0, len(region[1])):
            annot.append((region[1][ii]))
            annot.append((region[2][ii]))

        annotation_data = {
            "id": (name_mapping_dic[image_name] * 10000) + i,
            "category_id": region[0],
            "iscrowd": 0,
            "segmentation": [annot],
            "image_id": name_mapping_dic[image_name],
        }

        regions_data.append(annotation_data)
        i += 1

    return regions_data


def get_image_name(zipfile_name, name_mapping_dict):
    file_parts = zipfile_name.split('_')[1:3]
    name = file_parts[0] + '.' + file_parts[1]

    name = zipfile_name.replace('ANN_', '').split('_jpg')[0] + '.jpg'

    if name not in name_mapping_dict:
        name_mapping_dict[name] = len(name_mapping_dict) + 1

    return name, name_mapping_dict


def ensure_directory_existence(directory_path):
    if not os.path.isdir(directory_path):
        try:
            os.mkdir(directory_path)
        except OSError:
            print("Creation of the directory %s failed" % directory_path)
        return


def empty_directory(directory_path):
    for f in [f for f in os.listdir(directory_path)]:
        os.remove(os.path.join(directory_path, f))


def get_polygons_for_image(regions):
    regions_data = []
    
    for region in regions:
        annot = []

        for ii in range(0, len(region[1])):
            annot.append((int(region[1][ii]), int(region[2][ii])))

        regions_data.append((region[0], annot))

    return regions_data


def process_data(input_directory, output_directory, index_file='index.txt', force_load=False):
    # input_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    # images_directory = 'C:\\Projects\\tooth_damage_detection\\data\\annotator\\'
    output_directory = os.path.abspath(output_directory)
    input_directory = os.path.abspath(input_directory)

    colors = {
        "cat_1": (0, 128, 0),
        "cat_2": (255, 255, 0),
        "cat_3": (255, 0, 255),
        "cat_4": (255, 0 ,0),
        "cat_5": (0, 0, 255),
        "cat_6": (0, 0, 0)
    }

    masks_dir = os.path.join(output_directory, 'masks')
    images_dir = os.path.join(output_directory, 'images')
    tfrecord_dir = os.path.join(output_directory, 'tfrecord')

    ensure_directory_existence(output_directory)
    
    ensure_directory_existence(masks_dir)
    empty_directory(masks_dir)
    
    ensure_directory_existence(images_dir)
    empty_directory(images_dir)

    ensure_directory_existence(tfrecord_dir)
    empty_directory(tfrecord_dir)
    
    annotation_files = get_input_files(input_directory)

    name_mapping = dict()

    files = []

    for annotation_filename in annotation_files:
        print('opening... ' + annotation_filename)

        image_name, name_mappings = get_image_name(annotation_filename, name_mapping)
        name_mapping = name_mappings
        annotation_file_path = os.path.join(input_directory, annotation_filename)

        polygons = get_roi_files_from_zipfile(annotation_file_path, 'superpixel')

        regions = process_regions_of_interest(polygons)
        annotation_file_data = get_polygons_for_image(regions)

        image = cv2.imread(os.path.join(input_directory, image_name), -1)
        mask = np.ones(image.shape, dtype=np.uint8)
        mask[:,0:image.shape[1]] = (0, 255, 255) 

        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count

        for annotat in annotation_file_data:
            roi_corners = np.array([[annotat[1]]], dtype=np.int32)
            annot_color = colors['cat_' + str(annotat[0])]
            cv2.fillPoly(mask, roi_corners, color=annot_color)

        # cv2.imshow(winname = "image", mat = mask)
        # cv2.waitKey(delay = 0)

        cv2.imwrite(os.path.join(masks_dir, image_name.replace('.jpg', '.png')), mask)
        cv2.imwrite(os.path.join(images_dir, image_name), image)
        files.append(image_name.replace('.jpg', ''))

        print(str(len(regions)) + ' regions found')

    with open(index_file, 'w') as outfile:
        for img in files:
            outfile.writelines(img +'\n')

if __name__ == "__main__":
    process_data('C:\\Projects\\tooth_damage_detection_deeplab\\data\\annotator\\training\\', 'C:\\Projects\\tooth_damage_detection_deeplab\\data\\output\\')