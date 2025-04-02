from matplotlib.pyplot import plot
import torch
import cv2
import numpy as np
import sys
from pathlib import Path
import matplotlib.cm as cm
from os.path import join
import math
import networkx as nx
import os

from cirtorch_networks.imageretrievalnet import extract_ss, extract_ms

class LikelihoodEstimator:
    def __init__(self, vg=False, net=None, method = 'topk',top_k = 7, short_sim = 0.95, threshold = 0.7, cov_threshold=0.95, loftr_conf = 0.4, use_mlp=False):
        self.vg = vg
        self.top_k = top_k
        self.short_similarity = short_sim
        self.loftr_conf = loftr_conf
        self.method = method
        self.threshold = threshold
        self.cov_threshold = cov_threshold
        if threshold == 0.5:
            self.low_value = 0.2
        else:
            self.low_value = 0.15
        self.net = net
        self.use_mlp = use_mlp

    def extract_descriptor(self, net, image, ms, msp):
        if self.vg:
            # Extract descriptor as a column vector and normalize it
            descriptor = net(image).permute(1,0)
            descriptor = descriptor / descriptor.norm(dim=0)
            descriptor = descriptor.cpu().data.squeeze()
            return descriptor
        else:
            if len(ms) == 1 and ms[0] == 1:
                descriptor = extract_ss(net, image)
            else:
                descriptor = extract_ms(net, image, ms, msp)
            return descriptor

    def global_descriptor_similarity(self, incoming_descriptor, node_descriptor):
        # Compare descriptor depending of vg. If vg, use L2, if not, use cosine.
        if self.vg and not self.use_mlp:
            # Scale L2-norm to 0-1. A distance of 0 should give a similarity of 1, while
            # bigger distances should give smaller similarities, tending to 0.
            # distance = np.linalg.norm(incoming_descriptor.numpy() - node_descriptor.numpy(), axis=0)
            # similarity = 1 - min(distance, self.max_distance) / self.max_distance
            similarity = np.dot(incoming_descriptor.numpy().T, node_descriptor.numpy())
            return similarity
        elif self.vg and self.use_mlp:
            if len(incoming_descriptor.shape) == 1:
                incoming_descriptor = incoming_descriptor.unsqueeze(1)
            augmented_feats = (incoming_descriptor - node_descriptor).permute(1,0)
            sim = self.net(augmented_feats, compare=True, softmax=True).squeeze(0).cpu().numpy()
            return sim
        else:
            return np.dot(incoming_descriptor.numpy().T, node_descriptor.numpy())

    def estimate_similarities_regional(self, graph, descriptor):
        if graph.voting:
            return self.estimate_similarities_voting(graph, descriptor)
        else:
            return self.estimate_similarities_avg(graph, descriptor)
    
    def estimate_similarities_avg(self, graph, descriptor):
        similarities = np.zeros(len(graph.nodes))
        if len(descriptor.shape) == 1:
            descriptor = descriptor.unsqueeze(1)

        if graph.window is not None:
            window_search = list(nx.ego_graph(graph.nx_graph, graph.current_position, radius = graph.window).nodes())
        else:
            window_search = list(graph.nodes.keys())

        # Descriptor comes from regional node
        #for id, node in graph.nodes.items():
        for id in window_search:
            node = graph.nodes[id]
            # Each regional node is composed of different submaps (multi-node)
            # Regional similarity between each submap in the regional node and the new submap
            regional_similarity = np.zeros(len(node.covisible_nodes)) 
            for n, submap in enumerate(node.covisible_nodes):
                # Similarity between each incoming descriptor (from new submap) and the current submap (regional)
                similarity = np.zeros(descriptor.shape[1]) 
                for desc in range(0, descriptor.shape[1]):
                    # similarity[desc] = np.max(self.global_descriptor_similarity(descriptor[:, desc], submap.descriptor))
                    sim = self.global_descriptor_similarity(descriptor[:, desc], submap.descriptor)
                    n_desc = 3 #math.ceil(len(sim)*0.3)
                    ranked_sim = np.argsort(-sim, axis=0)[:n_desc]
                    similarity[desc] = np.mean(sim[ranked_sim])

                    # Save values for later visualization
                    for best_id in ranked_sim:
                        score = sim[best_id]
                        image_id = submap.image_paths[best_id]
                        graph.highest_similarities = self.add_score(graph.highest_similarities, image_id, score)


                regional_similarity[n] = np.mean(similarity) 

            # Regional similarity is the max of the similarities between the new submap and each submap in the regional node
            similarities[id] = np.amax(regional_similarity)

        return similarities
    
    def estimate_similarities_voting(self, graph, descriptor):
        votes = np.zeros(len(graph.nodes))
        if len(descriptor.shape) == 1:
            descriptor = descriptor.unsqueeze(1)

        if graph.window is not None:
            window_search = list(nx.ego_graph(graph.nx_graph, graph.current_position, radius = graph.window).nodes())
        else:
            window_search = list(graph.nodes.keys())

        # Descriptor comes from regional node
        #for id, node in graph.nodes.items():
        for id in window_search:
            node = graph.nodes[id]
            # Each regional node is composed of different submaps (multi-node)
            # Regional similarity between each submap in the regional node and the new submap
            regional_votes = np.zeros(len(node.covisible_nodes)) 
            for n, submap in enumerate(node.covisible_nodes):
                # Similarity between each incoming descriptor (from new submap) and the current submap (regional)
                voting = np.zeros(descriptor.shape[1]) 
                for desc in range(0, descriptor.shape[1]):
                    # similarity[desc] = np.max(self.global_descriptor_similarity(descriptor[:, desc], submap.descriptor))
                    sim = self.global_descriptor_similarity(descriptor[:, desc], submap.descriptor)
                    ranked_sim = np.argsort(-sim, axis=0)[:3]

                    # if node.id == 36:
                    #     print(f'## Node: {id} - Submap: {n} - Sim: {sim[ranked_sim[0]]} - Image name {submap.image_paths[ranked_sim[0]]}')

                    # Retrieve the ones above threshold and get percentage of votes
                    voting[desc] = np.sum(sim > self.threshold) / len(sim)

                    # Save values for later visualization
                    for best_id in ranked_sim:
                        score = sim[best_id]
                        image_id = submap.image_paths[best_id]
                        graph.highest_similarities = self.add_score(graph.highest_similarities, image_id, score)


                regional_votes[n] = np.mean(voting) 
                #TODO: check if this is the best way to combine votes

            # Regional similarity is the max of the similarities between the new submap and each submap in the regional node
            votes[id] = np.mean(regional_votes)

        return votes
    
    def check_intra_covisibility(self, submap, save=False):
        descriptor = submap.descriptor
        best_frame = -1
        best_sim = None
        best_indexes = []
        indexes = []
        # similarity = np.zeros(descriptor.shape[1]) 
        for desc in range(0, descriptor.shape[1]):
            sim = self.global_descriptor_similarity(descriptor[:, desc], descriptor)

            # Retrieve the ones above threshold
            indexes.append(np.where(sim > self.cov_threshold)[0])

            # Check if it's the best frame
            if len(indexes[-1]) > len(best_indexes):
                best_frame = desc
                best_sim = sim
                best_indexes = indexes[-1]

        # Print log
        print(f'Best frame: {best_frame} - Total frames: {descriptor.shape[1]} - Covisibility: {len(best_indexes)}')
        # Clean submap
        submap.filter_by_covisibility(best_indexes)
        return submap
    
    def add_score(self, scores_dict, item_id, score, max_size = 8):
        if item_id in scores_dict:
            # If the item already exists, update the score if it's higher
            if score > scores_dict[item_id]:
                scores_dict[item_id] = score
        elif len(scores_dict) < max_size:
            # If the list is not full, simply add the new item
            scores_dict[item_id] = score
        else:
            # If the list is full, find the item with the lowest score
            min_id = min(scores_dict, key=scores_dict.get)

            # Compare the lowest score with the new score
            if score > scores_dict[min_id]:
                # Replace the item with the lowest score
                del scores_dict[min_id]
                scores_dict[item_id] = score
        scores_dict = dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True))        
        return scores_dict
