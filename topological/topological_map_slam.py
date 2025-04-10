from collections import OrderedDict
import sys
from pathlib import Path
import numpy as np
import torch
import networkx as nx
import matplotlib.cm as cm
import os.path
import pickle

import cv2
import kornia as K
import kornia.feature as KF
from lightglue import LightGlue, SuperPoint

from settings import DATA_PATH, EVAL_PATH, ASSETS_PATH

from .node_objects import MultiNode, RegionalNode
from .matching import verify_node

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 20)
fontScale = 1
lineType = 2

class TopologicalMapSLAM:
    def __init__(self, conf, dimension, likelihood_estimator, window=None, vg=False):
        self.data_path = conf['data_path']
        self.graph_path = conf['graph_path']

        self.last_image_inserted = 0
        self.descriptor_dimension = (dimension, 1)

        self.nx_graph = nx.Graph()
        self.nodes = OrderedDict()
        self.n_connections = OrderedDict()
        self.current_position = -1
        self.state = "ok"
        self.last_inserted = None
        self.num_regional_nodes = 0

        self.default_image = cv2.resize(cv2.imread(f'{ASSETS_PATH}/noimages.jpg'), (250, 200))
        self.first_plot = True
        self.read_image = True
        self.most_similar = -1
        self.scores = []

        # Last inserted/updated nodes.
        self.last_nodes = []

        # verification
        if conf['verification'] is not None:
            verification = conf['verification']
            self.device = verification['device']
            # self.resize = conf['verification']['resize']
            self.extractor = SuperPoint(max_num_keypoints = verification['superpoint']['max_num_keypoints'], 
                                        detection_threshold = verification['superpoint']['detection_threshold']).eval().to(self.device)
            self.matcher = LightGlue(features=verification['lightglue']['features'], 
                                     depth_confidence=verification['lightglue']['depth_confidence'], 
                                     width_confidence=verification['lightglue']['width_confidence'], 
                                     filter_threshold=verification['lightglue']['filter_threshold']).eval().to(self.device)
            
            self.lightglue = True # 
            self.verification = True
            self.do_viz = False
        else:
            self.matcher = None
            self.lightglue = False
            self.verification = False
            self.do_viz = False

        self.likelihood_estimator = likelihood_estimator
        self.mapped_frames = []

        # Visualization
        self.highest_similarities = {}
        self.visualize_likelihood = {}
        self.in_window = []
        self.graph_updates = {}
        self.candidates = {}
        self.best_match = None
        self.localized = -1

        # Whether to use a small window to limit search
        self.window = window
        self.initial_window = window
        self.dynamic_window = conf['dynamic_window']
        
        # Comparison flags
        self.only_LG = conf['only_LG']
        self.only_similarity = conf['only_similarity']
        self.miccai = conf['miccai']
        self.single_candidate = conf['single_candidate']
        self.slam_verification = conf['slam_verification']

        # Thresholds
        self.sim_threshold = conf['sim_threshold']
        self.matches_threshold = conf['matches_threshold']
        self.voting_threshold = conf['voting_threshold']

        self.localizations = []

        # Likelihood method
        if self.voting_threshold > 0.0:
            self.voting = True
        else:
            self.voting = False

    # --- GRAPH MODIFICATION ---
    def initialize_graph_v2(self, graph_data, initial_localization = None):
        # Load nodes and images
        ids = graph_data['node_ids']
        images_paths = graph_data['images']
        n_frames = graph_data['n_frames']
        descriptors = graph_data['descriptors']
        edges = graph_data['edges']
        
        # Add nodes to graph
        for idx, node_id in enumerate(ids):
            first_node = MultiNode(descriptors[idx][0], images_paths[idx][0], n_frames[idx][0], id=0)
            self.create_regional_node(first_node, add_edge=False)
            print('Regional node created {}'.format(idx))
            if len(images_paths[idx]) > 1:
                for i in range(1, len(images_paths[idx])):
                    submap_node = MultiNode(descriptors[idx][i], images_paths[idx][i], n_frames[idx][i], id=0)
                    self.nodes[idx].covisible_nodes.append(submap_node)
                    print('Covisible node added to regional node {}'.format(idx))
                    print('Check ids: {} and {}'.format(self.nodes[idx].id, idx))

        # Add edges between nodes
        self.add_graph_edges(edges)

        # Assume we start at initial_node
        # if initial_localization is not None:
        #     init = initial_localization - ids[0]
        #     print('>> Starting at node {}'.format(init))

        # TODO: check this
        # self.current_position = len(ids) - 1
        self.current_position = 2
        print('>> Starting at node {}'.format(self.current_position))

    def initialize_graph(self, graph_data, initial_localization = None):
        # Load nodes and images
        ids = graph_data['node_ids']
        images_paths = graph_data['images']
        n_frames = graph_data['n_frames']
        descriptors = graph_data['descriptors']
        
        # Add nodes to graph
        for idx, node_id in enumerate(ids):
            first_node = MultiNode(descriptors[idx][0], images_paths[idx][0], n_frames[idx][0], id=0)
            self.create_regional_node(first_node)
            print('Regional node created {}'.format(idx))
            if len(images_paths[idx]) > 1:
                for i in range(1, len(images_paths[idx])):
                    submap_node = MultiNode(descriptors[idx][i], images_paths[idx][i], n_frames[idx][i], id=0)
                    self.nodes[idx].covisible_nodes.append(submap_node)
                    print('Covisible node added to regional node {}'.format(idx))
                    print('Check ids: {} and {}'.format(self.nodes[idx].id, idx))

        # TODO: Add edges between nodes

        # Assume we start at initial_node
        if initial_localization is not None:
            init = initial_localization - ids[0]
            print('>> Starting at node {}'.format(init))

        # TODO: check this
        self.current_position = len(ids) - 1

    
    def create_regional_node(self, submap_node, add_edge=True):
        # Initialize regional node to hold covisible submaps
        node = RegionalNode(submap_node, self.num_regional_nodes, probability=0.9)
        self.num_regional_nodes += 1

        previous_position = self.current_position
        if self.read_image:
            position = round(0.5 * len(submap_node.image_paths))
            submap_node.image = cv2.imread(submap_node.image_paths[position])
            submap_node.image = cv2.resize(submap_node.image, (250, 200))
            node_text = str(node.id) + ' - ' + str(len(submap_node.image_paths))
            cv2.putText(submap_node.image, node_text, (20, 50), font, fontScale, (10, 255, 10), lineType)
            cv2.putText(submap_node.image, str(submap_node.n_frame), (20, 80), font, 1, (10, 255, 10), 1)
            submap_node.regional_id = node.id

        print('Adding node {}'.format(node.id))
        if (previous_position != -1) and add_edge:
            self.nx_graph.add_node(node.id)
            self.nx_graph.add_edge(self.nodes[previous_position].id, node.id)
            print('Connection between {} and {}'.format(node.id, previous_position))

            self.nodes[node.id] = node
        else:
            # First node doesn't have initial edges
            self.nx_graph.add_node(node.id)
            self.nodes[node.id] = node

        self.last_inserted = node
        self.current_position = node.id

    def add_graph_edges(self, edges):
        for edge in edges:
            self.nx_graph.add_edge(edge[0], edge[1])

    def add_covisible_link(self, submap_node, covisible_id, slam=False):
        # Covisible link can be added to current regional node or to a distant one
        # If the latter, we need to create a traversability link
        previous_position = self.current_position
        if (covisible_id == previous_position) or self.state == "lost":
            # Covisible link to current regional node
            self.nodes[covisible_id].covisible_nodes.append(submap_node)
            print('Covisible node added to regional node {}'.format(covisible_id))
        else:
            # Covisible link to distant regional node. Create a traversability link too
            self.nx_graph.add_edge(previous_position, covisible_id)
            self.nodes[covisible_id].covisible_nodes.append(submap_node)
            print('Covisible node added to regional node {}'.format(covisible_id))
            print('Traversability link added between {} and {}'.format(previous_position, covisible_id))

        if slam:
            self.nodes[covisible_id].recently_updated = True

        submap_node.regional_id = covisible_id 
        self.current_position = covisible_id

    def increase_uncertainty(self):
        # Window size increase with uncertainty wrt the last position
        if self.window is not None:
            self.window += 1

    def update_position(self, id):
        # Update current position and reset window
        self.current_position = id
        self.window = self.initial_window

    # --- PROBABILITIES with SUBMAPS ---
    def localize_submap(self, submap, descriptor):

        max_sim_index = {}
        list_loop_id = []
        list_loop_found = []
        list_max_prob_index = []
        avg_scores = None
        for n in range(descriptor.shape[1]):
            desc = descriptor[:, n]
            self.p_kt_pred = self.prediction()
            loop_id, loop_found, self.most_similar, max_sim, max_prob_index = self.update_probabilities(desc, self.p_kt_pred, return_idx=True)

            # print(f'## Submap id {submap.id} - Node id {loop_id} - Max similarity {max_sim:.2f} - Most similar {self.most_similar}')
            # print(f'## Max probability index {max_prob_index[1]} - Max probability {max_prob_index[0]:.2f}')

            # Create list and append to dictionary (only for only_similarity flag)
            if self.most_similar in max_sim_index:
                max_sim_index[self.most_similar].append(max_sim)
            else:
                max_sim_index[self.most_similar] = [max_sim]

            # Add self.scores to avg_scores
            if avg_scores is None:
                avg_scores = self.scores
            else:
                avg_scores = np.add(avg_scores, self.scores)

            # TODO: break if loop found
            # list_loop_id.append(temp_loop_id)
            # list_loop_found.append(temp_loop_found)
            # list_max_prob_index.append(temp_max_prob_index)
            # max_prob_index = temp_max_prob_index

        # Get loop_id != -1 with max number ocurrences and the number of them

        # Set self.scores to the average of all scores
        self.scores = avg_scores / descriptor.shape[1] 
        # print('>> Average scores: {}'.format(self.scores))       

        # Run verification on the most probable node
        if self.verification:
            # max_prob = max_prob_index[0]
            # max_index = max_prob_index[1]
            max_matches = 0
            best_id = -1
            if self.only_LG == True:
                # We only use LighGlue for localization
                loop_found, best_id, saved_pair, max_matches = self.matching_verification(submap)
                
                if loop_found:
                    print('>> Verification successful with {} matches'.format(max_matches))
                    print('>> Best pair: {}'.format(saved_pair))
                    loop_id = best_id
            else:
                # We use both L network + LighGlue for localization
                lg_loop_found, lg_best_id, saved_pair, max_matches = self.matching_verification(submap)
                
                if lg_loop_found:
                    print('>> Verification successful with {} matches'.format(max_matches))
                    print('>> Best pair: {}'.format(saved_pair))
                    loop_id = lg_best_id
                    loop_found = True
                else:
                    # Average similarities for each node
                    avg_sim = {}
                    max_sim = 0
                    max_ocurrences = 0
                    max_index = -1
                    for key, value in max_sim_index.items():
                        sim = np.median(value)
                        ocurrences = len(value)
                        if sim > max_sim and ocurrences > max_ocurrences:
                            max_sim = sim
                            max_index = key
                            max_ocurrences = ocurrences
                            # WARNING: This does not consider the number of ocurrences i.e. with one is enough
                            # WE SHOULD ADD A CONSISTENCY CHECK

                    # Only use similarities to find loop
                    loop_found = (max_sim > self.sim_threshold) #(max_sim > 0.55)
                    loop_id = max_index
                    if loop_found:
                        print('>> Loop found with similarity {}'.format(max_sim))
                # # Verify against the most probable node and the current position
                # if max_index != self.current_position:
                #     nodes_verified = [max_index, self.current_position]
                # else:
                #     nodes_verified = [max_index]
                
                # for id in nodes_verified:
                #     map_node = self.nodes[id]
                #     n_matches, best_pair = verify_node(self.extractor, self.matcher, map_node, submap)
                #     if n_matches > self.matches_threshold:
                #         if n_matches > max_matches:
                #             max_matches = n_matches
                #             max_index = id
                #             print('>> Verification successful with {} matches'.format(n_matches))
                #             print('>> Best pair: {}'.format(best_pair))
                #             loop_found = True
                #             loop_id = max_index
                #     elif max_prob > 0.7:
                #         print('>> Verification failed but high probability')
                #         loop_found = True
                #         loop_id = max_index
                #     else:
                #         print('>> Loo was rejected')
                #         loop_found = False
                #         loop_id = -1
        elif self.only_similarity:
            # Average similarities for each node
            avg_sim = {}
            max_sim = 0
            max_ocurrences = 0
            max_index = -1
            for key, value in max_sim_index.items():
                sim = np.median(value)
                ocurrences = len(value)
                if sim > max_sim and ocurrences > max_ocurrences:
                    max_sim = sim
                    max_index = key
                    max_ocurrences = ocurrences
                    # WARNING: This does not consider the number of ocurrences i.e. with one is enough
                    # WE SHOULD ADD A CONSISTENCY CHECK
                
            # Only use similarities to find loop
            loop_found = (max_sim > self.sim_threshold) #(max_sim > 0.55)
            loop_id = max_index
            if loop_found:
                print('>> Loop found with similarity {}'.format(max_sim))
        elif self.voting:
            # Find the two nodes with higher votes in self.scores and their indices
            max_votes = np.argsort(-self.scores)[:2]
            print('>> Max votes: {}'.format(self.scores[max_votes[0]]))

            # Add LightGlue verification here

            # Check if the two nodes have enough votes
            if self.scores[max_votes[0]] > self.voting_threshold:
                loop_found = True
                loop_id = max_votes[0]
                print('>> Loop found with {} votes'.format(self.scores[max_votes[0]]))
            else:
                loop_found = False
                loop_id = -1

        if loop_found:
            print('Loop found in submap {}'.format(loop_id))
        else:
            print('No loop found')

        return loop_found, loop_id
    
    def localize_submap_v2(self, submap, descriptor):
        # Localize using the network and verify with LightGlue
        max_sim_index = {}
        avg_scores = None
        loop_found = False
        loop_id = -1
        for n in range(descriptor.shape[1]):
            # print(f'## Image being searched {submap.image_paths[n]}')
            desc = descriptor[:, n]
            scores, self.most_similar, max_sim = self.compare_submaps(desc)
            self.scores = scores

            # Create list and append to dictionary (only for only_similarity flag)
            if self.most_similar in max_sim_index:
                max_sim_index[self.most_similar].append(max_sim)
            else:
                max_sim_index[self.most_similar] = [max_sim]

            # Add self.scores to avg_scores
            if avg_scores is None:
                avg_scores = self.scores
            else:
                avg_scores = np.add(avg_scores, self.scores)

        # Set self.scores to the average of all scores
        self.scores = avg_scores / descriptor.shape[1]       

        # We have 4 modes: similarity, voting, only_LG and miccai (similarity OR LG)
        if self.only_LG:
            max_matches = 0
            best_id = -1
            # We only use LighGlue for localization
            loop_found, best_id, saved_pair, max_matches = self.matching_verification(submap)
            
            if loop_found:
                print('>> Verification successful with {} matches'.format(max_matches))
                print('>> Best pair: {}, {}'.format(saved_pair[0], saved_pair[1]))
                loop_id = best_id
        elif self.only_similarity:
            # Average similarities for each node
            avg_sim = {}
            max_sim = 0
            max_ocurrences = 0
            max_index = -1
            # print(max_sim_index)
            for key, value in max_sim_index.items():
                sim = np.median(value)
                ocurrences = len(value)
                print(f'Node: {key}, sim: {sim}, ocurrences:{ocurrences}')
                if sim > max_sim and ocurrences > max_ocurrences:
                    max_sim = sim
                    max_index = key
                    max_ocurrences = ocurrences
                    # WARNING: This does not consider the number of ocurrences i.e. with one is enough
                    # WE SHOULD ADD A CONSISTENCY CHECK
                
            # Only use similarities to find loop
            loop_found = (max_sim > self.sim_threshold) #(max_sim > 0.55)
            loop_id = max_index
            if loop_found:
                print('>> Loop found with similarity {}'.format(max_sim))
        elif self.miccai:
            # We use both L network + LighGlue for localization
            lg_loop_found, lg_best_id, saved_pair, max_matches = self.matching_verification(submap)
            
            if lg_loop_found:
                print('>> Verification successful with {} matches'.format(max_matches))
                print('>> Best pair: {}, {}'.format(saved_pair[0], saved_pair[1]))
                loop_id = lg_best_id
                loop_found = True
            else:
                # Average similarities for each node
                avg_sim = {}
                max_sim = 0
                max_ocurrences = 0
                max_index = -1
                for key, value in max_sim_index.items():
                    sim = np.median(value)
                    ocurrences = len(value)
                    if sim > max_sim and ocurrences > max_ocurrences:
                        max_sim = sim
                        max_index = key
                        max_ocurrences = ocurrences
                        # WARNING: This does not consider the number of ocurrences i.e. with one is enough
                        # WE SHOULD ADD A CONSISTENCY CHECK

                # Only use similarities to find loop
                loop_found = (max_sim > self.sim_threshold) #(max_sim > 0.55)
                loop_id = max_index
                if loop_found:
                    print('>> Loop found with similarity {}'.format(max_sim))
                
        elif self.voting:
            # Find the two nodes with higher votes in self.scores and their indices
            max_votes = np.argsort(-self.scores)
            # print('>> Max votes: {}'.format(self.scores[max_votes[0]]))

            # Round the first number in max votes to 2 decimals and print
            print('>> Max votes: {}'.format(round(self.scores[max_votes[0]], 2)))

            single_candidate = self.single_candidate

            if single_candidate:
                # We only consider the node with the highest votes
                candidate = max_votes[0]

                # Check if the two nodes have enough votes
                if self.scores[candidate] > self.voting_threshold:
                    if self.lightglue and self.slam_verification is False:
                        lg_loop_found, saved_pair, max_matches = self.match_candidate(submap, candidate)
                        if lg_loop_found:
                            print('>> Verification successful with {} matches and {} votes'.format(max_matches, round(self.scores[candidate],2)))
                            print('>> Best pair: {}, {}'.format(saved_pair[0], saved_pair[1]))
                            loop_found = True
                            loop_id = candidate
                        else:
                            print('>> Candidate was not accepted with {} matches and {} votes'.format(max_matches, round(self.scores[candidate],2)))
                    else:
                        loop_found = True
                        loop_id = candidate
                        print('>> Loop found with {} votes in node {}'.format(round(self.scores[candidate],2), candidate))
            else:
                # We consider all nodes with the highest votes
                candidates = max_votes[:5]  

                # Store the candidates for visualization
                self.candidates = {}
                self.best_match = None
                for candidate in candidates:
                    self.candidates[candidate] = round(self.scores[candidate],2)

                # Check if the two nodes have enough votes
                for candidate in candidates:
                    if self.scores[candidate] > self.voting_threshold:
                        if self.lightglue and self.slam_verification is False:
                            lg_loop_found, saved_pair, max_matches = self.match_candidate(submap, candidate)
                            if lg_loop_found:
                                print('>> Verification successful with {} matches and {} votes'.format(max_matches, round(self.scores[candidate],2)))
                                print('>> Best pair: {}, {}'.format(saved_pair[0], saved_pair[1]))
                                self.best_match = saved_pair + (candidate,)
                                loop_found = True
                                loop_id = candidate
                                break
                            else:
                                print('>> Candidate was not accepted with {} matches and {} votes'.format(max_matches, round(self.scores[candidate],2)))
                        else:
                            loop_found = True
                            loop_id = candidate
                            print('>> Loop found with {} votes in node {}'.format(round(self.scores[candidate],2), candidate))
                            break
                    else:
                        print('>> Candidate was not accepted with {} votes'.format(round(self.scores[candidate],2)))
                        break # Candidates are sorted, so we can break here

        # If the loop was found in a recently updated candidate, run an additional verification with lightglue (if it was not already done)
        # This is only used when doing SLAM in a second sequence
        if loop_found and self.nodes[loop_id].recently_updated and self.slam_verification:
            lg_loop_found, saved_pair, max_matches = self.match_candidate(submap, loop_id)

            if lg_loop_found:
                print('>> Additional verification successful with {} matches'.format(max_matches))
                print('>> Best pair: {}, {}'.format(saved_pair[0], saved_pair[1]))
            else:
                print('>> Candidate was not accepted with {} matches'.format(max_matches))
                loop_id = -1
                loop_found = False    

        if loop_found:
            print('Loop found in submap {}'.format(loop_id))
        else:
            print('No loop found')
            loop_id = -1

        return loop_found, loop_id
    
    def compare_submaps(self, descriptor):
        scores = self.likelihood_estimator.estimate_similarities_regional(self, descriptor)

        most_similar = np.argmax(scores)
        max_sim = scores[most_similar]

        return scores, most_similar, max_sim
    
    def matching_verification(self, submap):
        # Use LightGlue to verify against certain nodes
        loop_found = False
        best_id = -1
        saved_pair = None
        max_matches = 0
        if self.window is not None:
            window_search = list(nx.ego_graph(self.nx_graph, self.current_position, radius = self.window).nodes())
        else:
            window_search = list(self.nodes.keys())
        # sub_window_nodes = list(sub_window.nodes())
        # sub_window = nx.ego_graph(self.nx_graph, self.current_position, radius = self.window)
        print('>> Verifying against {} nodes'.format(window_search))
        for id in window_search:
            node = self.nodes[id]
            n_matches, best_pair = verify_node(self.extractor, self.matcher, node, submap, exhaustive=True)
            if n_matches > self.matches_threshold:
                max_matches = n_matches
                saved_pair = best_pair
                loop_found = True
                best_id = id
                break

        return loop_found, best_id, saved_pair, max_matches
    
    def match_candidate(self, submap, candidate):
        # Use LightGlue to verify against the candidate node
        node = self.nodes[candidate]
        n_matches, best_pair = verify_node(self.extractor, self.matcher, node, submap, exhaustive=True)
        return n_matches > self.matches_threshold, best_pair, n_matches
    
    def save_reuse_results(self, sequence, experiment_name = None):
        output_path = EVAL_PATH
        if experiment_name is not None:
            output_path = os.path.join(output_path, experiment_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        # Also save a .txt with the format:
        # node_id: ids of covisible nodes
        text_filename = os.path.join(output_path, f'{sequence}_reuse_results.txt')
        graph_updates = os.path.join(output_path, f'{sequence}_reuse_updates.pkl')

        # Delete if already exists
        if os.path.exists(text_filename):
            os.remove(text_filename)

        for submap_id, loc_id in self.localizations:
            with open(text_filename, 'a') as f:
                # Format:
                # node_id: cov_id0 cov_id1 cov_id2 ...
                f.write(f'{submap_id}: {loc_id}\n')

        with open(graph_updates, "wb") as f:
            pickle.dump(self.graph_updates, f)

    def save_results(self, sequence, experiment_name = None):
        output_path = EVAL_PATH
        if experiment_name is not None:
            output_path = os.path.join(output_path, experiment_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        # Also save a .txt with the format:
        # node_id: ids of covisible nodes
        text_filename = os.path.join(output_path, f'{sequence}_results.txt')

        # Delete if already exists
        if os.path.exists(text_filename):
            os.remove(text_filename)

        for submap_id, loc_id in self.localizations:
            with open(text_filename, 'a') as f:
                # Format:
                # node_id: cov_id0 cov_id1 cov_id2 ...
                f.write(f'{submap_id}: {loc_id}\n')

    def save_runtime(self, avg_runtime, experiment_name = None, folder=None):
        runtime_file = EVAL_PATH + "runtime.txt"
        if folder is not None:
            runtime_file = EVAL_PATH + "runtime.txt"
        if avg_runtime > 60:
            minutes = int(avg_runtime // 60)
            seconds = avg_runtime % 60
            runtime = f"{minutes} minutes and {seconds:.2f} seconds"
        else:
            runtime = f"{avg_runtime:.2f} seconds"

        with open(runtime_file, "a") as r:
            r.write(f'{experiment_name}: {runtime}\n')
        print("Runtime added to runtime.txt")


    def save_graph(self, sequence, experiment_name = None):
        # graph_pkl = os.path.join(self.graph_path)
        ids = []
        n_frames = []
        images = []
        descriptors = []
        output_path = EVAL_PATH
        if experiment_name is not None:
            output_path = os.path.join(output_path, experiment_name)
            graph_pkl = os.path.join(output_path, self.graph_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        # Also save a .txt with the format:
        # node_id: ids of covisible nodes
        text_filename = os.path.join(output_path, f'{sequence}_results.txt')
        graph_updates = os.path.join(output_path, f'{sequence}_updates.pkl')

        # Delete if already exists
        if os.path.exists(text_filename):
            os.remove(text_filename)
        for node_id, node in self.nodes.items():
            ids.append(node_id)
            n = []
            im = []
            des = []
            cov_ids = []
            # Every regional node has a list of covisible nodes
            for covisible in node.covisible_nodes:
                n.append(covisible.n_frame)
                im.append(covisible.image_paths)
                des.append(covisible.descriptor)
                cov_ids.append(covisible.id)

            n_frames.append(n)
            images.append(im)
            descriptors.append(des)

            with open(text_filename, 'a') as f:
                # Format:
                # node_id: cov_id0 cov_id1 cov_id2 ...
                f.write(f'{node_id}: {cov_ids}\n')

        # Save edges
        edges = list(self.nx_graph.edges)

        print('>> {}: Saving graph pkl...')
        data = {'node_ids': ids, 'n_frames': n_frames, 'images': images, 'descriptors': descriptors, 'edges': edges}

        # with open(graph_pkl, "wb") as f:
        #     pickle.dump(data, f)

        with open(graph_updates, "wb") as f:
            pickle.dump(self.graph_updates, f)

    def save_update(self, idx):
        # Save the current state of the graph
        ids = []
        counter = []
        image_names = []
        
        for node_id, node in self.nodes.items():
            ids.append(node_id)
            counter.append(len(node.covisible_nodes))
            image_names.append(node.covisible_nodes[0].image_paths[0])

        candidates = self.candidates
        best_match = self.best_match

        state = {'nodes': ids, 'edges': list(self.nx_graph.edges), 'counter': counter, 
                 'image_names': image_names, 'current_position': self.current_position,
                 'candidates': candidates, 'best_match': best_match}
        self.graph_updates[idx] = state

        print('>> {}: Graph updated'.format(idx))

    def save_update_reuse(self, idx):
        # Save the current state of the graph
        ids = []
        counter = []
        image_names = []
        
        for node_id, node in self.nodes.items():
            ids.append(node_id)
            counter.append(len(node.covisible_nodes))
            image_names.append(node.covisible_nodes[0].image_paths[0])

        candidates = self.candidates
        localized = self.localized
        localized_image_list = []
        if localized != -1:
            node_loc = self.nodes[localized]
            # Get a list of image paths from the localized node
            for i in range(len(node_loc.covisible_nodes)-1):
                covisible = node_loc.covisible_nodes[i]
                for img_path in covisible.image_paths:
                    localized_image_list.append(img_path)

        state = {'nodes': ids, 'edges': list(self.nx_graph.edges), 'counter': counter, 
                 'image_names': image_names, 'localized': localized, 'candidates': candidates,
                 'localized_image_list': localized_image_list, 'loc_state': self.state}
        self.graph_updates[idx] = state

        print('>> {}: Graph updated'.format(idx))
