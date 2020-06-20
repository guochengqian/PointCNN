#!/usr/bin/python()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np

"""
prepare_s3dis_label.py
load the xyz + color features of each objects inside each room 
and then save features `xyzrgb.npy` into the outdir for each room. 
.labels inside each room of the outdir is the marker, indicating we've processed this room
"""

# PointCNN use pillar-based methods. dataset is the aligned version.
# KPConv use the unaligned version. ball sample based.
# pointnet, DeepGCN use the preprocessed h5 file. (a subsample of the whole dataset,
# which does not convery all the points)

DEFAULT_DATA_DIR = '/data/3D/Stanford3dDataset_v1.2_Aligned_Version'
DEFAULT_OUTPUT_DIR = '/data/3D/s3dis_aligned/prepare_label_rgb'

p = argparse.ArgumentParser()
p.add_argument(
    "-d", "--data", dest='data_dir',
    default=DEFAULT_DATA_DIR,
    help="Path to S3DIS data (default is %s)" % DEFAULT_DATA_DIR)
p.add_argument(
    "-f", "--folder", dest='output_dir',
    default=DEFAULT_OUTPUT_DIR,
    help="Folder to write labels (default is %s)" % DEFAULT_OUTPUT_DIR)

args = p.parse_args()

object_dict = {
            'clutter':   0,
            'ceiling':   1,
            'floor':     2,
            'wall':      3,
            'beam':      4,
            'column':    5,
            'door':      6,
            'window':    7,
            'table':     8,
            'chair':     9,
            'sofa':     10,
            'bookcase': 11,
            'board':    12}

# subfolders of six areas.
path_dir_areas = os.listdir(args.data_dir)

for area in path_dir_areas:
    # for each are (6 areas in total).
    path_area = os.path.join(args.data_dir, area)
    if not os.path.isdir(path_area):
        continue

    # rooms in each area.
    path_dir_rooms = os.listdir(path_area)

    for room in path_dir_rooms:
        path_annotations = os.path.join(args.data_dir, area, room, "Annotations")
        if not os.path.isdir(path_annotations):
            continue
        print(path_annotations)

        # check the existence of labels.
        path_prepare_label = os.path.join(args.output_dir, area, room)
        if os.path.exists(os.path.join(path_prepare_label, ".labels")):
            print("%s already processed, skipping" % path_prepare_label)
            continue

        # prepare labels.
        xyz_room = np.zeros((1, 6)) # xyz + color
        label_room = np.zeros((1, 1))
        # make store directories
        if not os.path.exists(path_prepare_label):
            os.makedirs(path_prepare_label)

        # ====> load the annotations for each object inside the room
        path_objects = os.listdir(path_annotations)
        for obj in path_objects:
            object_key = obj.split("_", 1)[0]   # name of the object
            try:
                val = object_dict[object_key]
            except KeyError:
                continue
            print("%s/%s" % (room, obj[:-4]))

            xyz_object_path = os.path.join(path_annotations, obj)
            try:
                xyz_object = np.loadtxt(xyz_object_path)[:, :]  # (N,6)
            except ValueError as e:
                print("ERROR: cannot load %s: %s" % (xyz_object_path, e))
                continue
            label_object = np.tile(val, (xyz_object.shape[0], 1))  # (N,1)
            xyz_room = np.vstack((xyz_room, xyz_object))
            label_room = np.vstack((label_room, label_object))

        xyz_room = np.delete(xyz_room, [0], 0)
        label_room = np.delete(label_room, [0], 0)

        np.save(path_prepare_label+"/xyzrgb.npy", xyz_room)
        np.save(path_prepare_label+"/label.npy", label_room)

        # Marker indicating we've processed this room
        open(os.path.join(path_prepare_label, ".labels"), "w").close()
