from audioop import mul
from numpy import empty
import torch
import numpy as np
import cv2
import os

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 20)
fontScale = 1
lineType = 2

class Node:
    def __init__(self, descriptor, image_path, n_frame, id, probability = 0.0):
        self.descriptor = descriptor
        self.image_path = image_path
        self.n_frame = n_frame
        self.id = id
        self.probability = probability

        self.image = None
        self.keypoints_data = None

    def add_more_views(self, new_descriptor):
        self.descriptor = torch.cat((self.descriptor, new_descriptor), dim = 1)

class MultiNode:
    def __init__(self, descriptor, image_paths, n_frame, id, probability = 0.0):
        self.descriptor = descriptor
        if isinstance(image_paths, list):
            self.image_paths = image_paths
            self.frames_in_node = len(image_paths)
        else:
            self.image_paths = [image_paths]
            self.frames_in_node = 1
        self.n_frame = n_frame
        self.id = id
        self.probability = probability
        self.similarity = None
        self.image = None
        self.keypoints_data = None
        self.regional_id = None
        self.recently_added = False

    def add_more_views(self, new_descriptor, image):
        self.descriptor = torch.cat((self.descriptor, new_descriptor), dim = 1)
        if isinstance(image, list):
            self.image_paths.extend(image)
            self.frames_in_node += len(image)
        else:
            self.image_paths.append(image)
            self.frames_in_node += 1

    def number_of_frames(self):
        return self.frames_in_node
    
    def filter_by_covisibility(self, indexes):
        self.descriptor = self.descriptor[:, indexes]
        self.image_paths = [self.image_paths[i] for i in indexes]
        self.frames_in_node = len(indexes)
        self.n_frame = os.path.basename(self.image_paths[0]).replace('.png', '').zfill(5)
    
class RegionalNode:
    def __init__(self, multi_node, id, probability = 0.5):
        self.covisible_nodes = [multi_node]
        self.id = id
        self.probability = probability
        self.recently_updated = False