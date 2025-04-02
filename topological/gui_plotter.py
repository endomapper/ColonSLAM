import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2
import numpy as np
import math
from os.path import join
import os
from settings import DATA_PATH, EVAL_PATH, ASSETS_PATH

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 20)
fontScale = 1
lineType = 2

class GUIPlotter:
    def __init__(self, num_images = 100, max_width = 3250, max_height = 1800, frame_size = (300,250), sequence = '027', experiment_name = None):
        self.num_images = num_images #int(num_images * 0.7)
        self.max_width = max_width
        self.max_height = max_height
        self.frame_size = frame_size
        self.sequence = sequence
        self.experiment_name = experiment_name

        self.rows, self.cols, self.width, self.height = self.calculate_best_grid()

        # Load default images
        self.default_image = cv2.resize(cv2.imread(ASSETS_PATH + '/default.jpg'), frame_size)
        self.default_image_mapping = cv2.resize(cv2.imread(ASSETS_PATH + '/default.jpg'), (self.width, self.height))
        self.dustbin_images = []
        self.previous_frames = []

        self.count = 0


    def calculate_best_grid(self, init_cols=13, init_width=250, init_height=200, aspect_ratio=1.25):
        # Do initial estimation
        height = init_height
        width = init_width
        cols = init_cols
        rows = math.ceil(self.num_images / cols)

        height_occupied = rows * height
        if height_occupied > self.max_height:
            print("The initial grid does not fit in the given height")
            # Reduce size aggresively
            while height_occupied > self.max_height:
                height = height - 10
                width = height * aspect_ratio
                height_occupied = rows * height
            #assert False, "The initial grid does not fit in the given height"

        while (height_occupied < self.max_height):
            # Do another estimation by increasing the size of the images
            width = width + 10
            height = width / aspect_ratio

            # Check if with the current size we need to reduce cols
            width_occupied = cols * width
            while width_occupied > self.max_width:
                cols = cols - 1
                width_occupied = cols * width

            # Calculate the number of rows
            rows = math.ceil(self.num_images / cols)
            height_occupied = rows * height

        if height_occupied < self.max_height:
            height = height + (self.max_height - height_occupied) / rows
            width = height * aspect_ratio

        return rows, cols, int(width), int(height)

    def plot_topological_slam(self, graph, current_node, frame=None, loop_id=-1, plot=False, keyboard=False, wait = 100):
        if not plot:
            return
        # print('>> Plotting SLAM iteration...')
        nodes = list(graph.nodes.values())

        images_original = [node.covisible_nodes[0].image for node in graph.nodes.values()]
        images = []

        if graph.window is not None:
            window_search = list(nx.ego_graph(graph.nx_graph, graph.current_position, radius = graph.window).nodes())
        else:
            window_search = list(graph.nodes.keys())

        for i, image in enumerate(images_original):
            #alpha = min(probabilities[i]*2, 1.0) 
            image = cv2.resize(image.copy(),(self.width, self.height))
            prob = graph.scores[i]
            gamma = min(prob + 0.2, 1.0)
            if i not in window_search:
                gamma = 0.2
            else:
                gamma += 0.5

            # print('Window search: ', window_search)
            alpha = prob
            # alpha = 1.0 # plot map without transparency
            blank = np.zeros_like(image)
            # Number of nodes in regional node
            num_nodes = len(graph.nodes[i].covisible_nodes)
            prob_text = str(round(prob, 2))
            if i != loop_id:
                jet_color = cv2.applyColorMap(np.array([[[alpha*255]]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
                color_rect = (int(jet_color[0]), int(jet_color[1]), int(jet_color[2]))
                image = cv2.rectangle(image.copy(), (0, 0), (self.width-1, self.height-1), color_rect, 20)
                # image = image.copy()
                blended = cv2.addWeighted(image, gamma, blank, 1-gamma, 0)
                cv2.putText(blended, str(num_nodes), (10, 140), font, 2, (10, 255, 10), 2)
                cv2.putText(blended, prob_text, (70, 140), font, fontScale, color_rect, 2)
                images.append(blended)
            else:
                color_rect = (int(254), int(254), int(254))
                image = cv2.rectangle(image.copy(), (0, 0), (self.width-1, self.height-1), (245,71,201), 20)
                cv2.putText(image, 'LOOP', (60, 50), font, fontScale, color_rect, lineType)
                cv2.putText(image, str(num_nodes), (10, 140), font, 2, (10, 255, 10), 2)
                cv2.putText(image, prob_text, (70, 140), font, fontScale, color_rect, 2)
                # image = image.copy()
                images.append(image)

        while len(images) < self.num_images:
            images.append(self.default_image_mapping)

        n_rows = self.rows
        rows = []
        for n in range(0, n_rows):
            if self.cols*(n+1) > len(images):
                end = len(images)
                img_h = cv2.hconcat(images[self.cols*n:end])
                if len(rows) < 1:
                    rows.append(img_h)
                else:
                    height, width, channels = rows[0].shape
                    mod_img = np.zeros((height, width, 3), np.uint8)
                    border_width = (width - img_h.shape[1]) // 2
                    mod_img = cv2.copyMakeBorder(img_h, 0, 0, border_width, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    rows.append(mod_img)
            else:
                end = self.cols*(n+1)
                img_h = cv2.hconcat(images[self.cols*n:end])
                rows.append(img_h)

        img_final = cv2.vconcat(rows)
        
        # Create GUI shape around map image
        height, width, channels = img_final.shape
        top = 50
        if len(self.dustbin_images) > 0:
            bottom = self.height + 50
        else:
            bottom = 50
        left = 30
        right = 900
        img_final = cv2.copyMakeBorder(img_final, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(40, 40, 40)) #map
        cv2.putText(img_final, 'CURRENT MAP', (int(width/2) - 80, 35), font, fontScale, (10, 255, 10), lineType)

        # Plot current node or frame
        if current_node is not None:
            frame = self.plot_cuda_sift_node(current_node)
            #frame = cv2.resize(frame, (200, 1500))
            frame_height, frame_width, _ = frame.shape
            img_final[100:100+frame_height, 40+width:40+width+frame_width, :] = frame # current frame
            cv2.putText(img_final, 'CURRENT SUBMAP', (width+int(frame_width/2) - 80, 80), font, fontScale, (10, 255, 10), lineType)
        else:
            state = 'Normal' if not graph.is_dustbin else 'Dustbin'
            frame = self.gui_plot_current_frame(frame, state, plot)
            frame_height, frame_width, _ = frame.shape
            img_final[100:100+frame_height, 40+width:40+width+frame_width, :] = frame # current frame
            cv2.putText(img_final, 'CURRENT FRAME', (width+int(frame_width/2) - 80, 80), font, fontScale, (10, 255, 10), lineType)

        # Plot highest similarity
        # Check if highest similarity is empty
        if len(graph.highest_similarities) > 0:
            frame_highest = self.plot_highest_similarity(graph.highest_similarities)
            #frame = cv2.resize(frame, (200, 1500))
            frame_highest_height, frame_highest_width, _ = frame_highest.shape
            # print(frame_highest.shape)
            img_final[100:100+frame_highest_height, 50+width+frame_width:50+width+frame_width + frame_highest_width, :] = frame_highest # current frame
            cv2.putText(img_final, 'MOST SIMILAR', (width+frame_width+int(frame_highest_width/2) - 80, 80), font, fontScale, (10, 255, 10), lineType)

            # Reset dictionary
            graph.highest_similarities = {}

        experiment_folder = f'{EVAL_PATH}/{self.experiment_name}'
        sequence_folder = f'{EVAL_PATH}/{self.experiment_name}/{self.sequence}'
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
        if not os.path.exists(sequence_folder):
            os.makedirs(sequence_folder)
        filename = join(sequence_folder, str(self.count).zfill(6) + '.png')
        cv2.imwrite(filename, cv2.resize(img_final, (1875, 800) ))
        self.count += 1


    def plot_map_node(self, graph, node_id):
        nodes = list(graph.nodes.values())
        covisible_node = nodes[node_id].covisible_nodes

        for node in covisible_node:
            # Read all image for the submap
            submap_idx_str = f"Node {node.id}"
            submap_images = [cv2.resize(cv2.imread(img), (150,100)) for img in node.image_paths]

            # subnode_img = cv2.vconcat(rows)
            subnode_img = self.plot_double_column(submap_images, frames_per_row = 2, max_images = 14)
            cv2.imshow(submap_idx_str, subnode_img)
            cv2.waitKey(100)

        cv2.waitKey(0)

        # Destroy windows
        for node in covisible_node:
            submap_idx_str = f"Node {node.id}"
            cv2.destroyWindow(submap_idx_str)

    def gui_plot_current_frame(self, image, state, plot = True):
        if not plot:
            return
        # print(f'>> Plotting current frame... {state}')
        color = (10, 255, 10) if state == 'Normal' else (255, 10, 10)
        current_image = cv2.imread(image)
        current_image = cv2.resize(current_image, self.frame_size)
        cv2.putText(current_image, state, (20, 40), font, fontScale, color, lineType)
        return current_image
    
    def plot_vertical_node(self, node):
        current_node = []
        if node is not None:
            for i, frame in enumerate(node.image_paths):
                current_image = cv2.imread(frame)
                current_image = cv2.resize(current_image, self.frame_size)
                current_node.append(current_image)

        while len(current_node) < 4:
            current_node.append(self.default_image)

        img_final = cv2.vconcat(current_node)
        return img_final
    
    def plot_cuda_sift_node(self, node):
        current_node = [cv2.resize(cv2.imread(img), (250,200)) for img in node.image_paths]
        img_final = self.plot_double_column(current_node, frames_per_row = 2, max_images = 14)

        return img_final
    
    def plot_highest_similarity(self, highest_similarity):
        # current_node = [cv2.resize(cv2.imread(img), (250,200)) for img in highest_similarity.keys()]
        current_node = []
        for id, value in highest_similarity.items():
            img = cv2.resize(cv2.imread(id), (250,200))
            # print(id)
            node_id = id.split('/')[-3]
            color = (10, 255, 10)
            text = str(round(value, 2)) + ' - ' + node_id
            cv2.putText(img, text, (20, 40), font, fontScale, color, lineType)
            current_node.append(img)
        img_final = cv2.vconcat(current_node)

        return img_final
    
    def plot_double_column(self, image_list, frames_per_row, max_images):

        # If image_list has more than max_images, reduce to max_images uniformly distributed images
        if len(image_list) > max_images:
            image_list = image_list[::math.ceil(len(image_list)/max_images)]

        n_rows = math.ceil(len(image_list) / frames_per_row)
        rows = []
        for n in range(0, n_rows):
            if frames_per_row*(n+1) > len(image_list):
                end = len(image_list)
                img_h = cv2.hconcat(image_list[frames_per_row*n:end])
                if len(rows) < 1:
                    rows.append(img_h)
                else:
                    height, width, channels = rows[0].shape
                    mod_img = np.zeros((height, width, 3), np.uint8)
                    border_width = (width - img_h.shape[1]) // 2
                    mod_img = cv2.copyMakeBorder(img_h, 0, 0, border_width, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    rows.append(mod_img)
            else:
                end = frames_per_row*(n+1)
                img_h = cv2.hconcat(image_list[frames_per_row*n:end])
                rows.append(img_h)
        
        image = cv2.vconcat(rows)
        return image
    
    def plot_previous_frames(self):
        previous = []
        for i, frame in enumerate(self.previous_frames):
            previous.append(frame)

        # Reverse list
        previous = previous[::-1]

        while len(previous) < 4:
            previous.append(self.default_image)

        img_final = cv2.vconcat(previous)
        return img_final