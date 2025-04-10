"""
Main functions for topological SLAM and map reuse.
"""

from asyncore import loop
from datetime import datetime
import logging
import numpy as np
import torch
import cv2

import time
from datasets.genericdataset import ImagesFromList
from datasets.vg_genericdataset import ImagesFromListVG
from .node_objects import MultiNode
from .topological_map_slam import TopologicalMapSLAM
from .gui_plotter import GUIPlotter
from .likelihood import LikelihoodEstimator
import matplotlib.pyplot as plt
import os

def topometric_slam(conf, net, submaps, image_size, output_dim, transform, vg=False, plot=False, ms=[1], msp=1):
    """ Process topological nodes coming from an external SLAM algorithm and intermediate frames between them.
        Every node is added to the graph, or fused to an existing node if a loop closure is found.
        Intermediate frames are used to update the localization probability of the next node.
    """
    # Create likelihood estimator
    likelihood_estimator = LikelihoodEstimator(vg, net=net, threshold=conf['sim_threshold'], cov_threshold=conf['cov_threshold'], use_mlp=conf['use_mlp'])

    # Create graph and get conf values
    graph = TopologicalMapSLAM(conf, output_dim, likelihood_estimator=likelihood_estimator, window=conf['window_size'], vg=vg)
    gui = GUIPlotter(len(submaps), sequence=conf['sequence'], experiment_name=conf['experiment_name'])

    logging.info('Starting a new map...')
    last_node_inserted = None 
     
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    
    loop_id = -1
    plot_frames = False
    plot_nodes = False

    # Iterate through every node in nodes
    for submap_idx, node in enumerate(submaps):
        if submap_idx < conf['start']:
            print('>> Skipping node {}...'.format(submap_idx))
            continue

        print('>> Processing node {} of {}...'.format(submap_idx, len(submaps)))

        # 1. Now process the node itself
        node_loader = torch.utils.data.DataLoader(
                ImagesFromListVG(root='', images=node, imsize=image_size, transform=transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
        
        # Get descriptors for every image in the node
        with torch.no_grad():
            submap_node = None
            for img_idx, input in enumerate(node_loader):
                input = input.cuda()
                image_path = submaps[submap_idx][img_idx]

                # Extract descriptor
                descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)
                descriptor = descriptor.view(descriptor.shape[0], 1)

                if submap_node is None:
                    n_frame = os.path.basename(image_path).replace('.png', '').zfill(5)
                    submap_node = MultiNode(descriptor, image_path, n_frame, submap_idx, probability = 0.0)
                else:
                    submap_node.add_more_views(descriptor, image_path)

            if conf['filter']:
                submap_node = likelihood_estimator.check_intra_covisibility(submap_node, save=False)

            # Node is finished, now we should add covisibility or traversability links
            if last_node_inserted is None:
                # First node: add it to the graph
                graph.create_regional_node(submap_node)
                last_node_inserted = graph.last_inserted
            else:
                # Graph is already initialized, decide which link to insert
                localization_status, loop_id = graph.localize_submap_v2(submap_node, submap_node.descriptor)
                # Plot graph
                gui.plot_topological_slam(graph, current_node=submap_node, frame=None, loop_id=loop_id, plot = plot_nodes, keyboard = False, wait = 300)
                # gui.plot_likelihood(graph)
                # localization_status = False
                if localization_status:
                    # We have a localization, add covisibility link (and maybe traversability link)
                    print('>> Localization found in node {}!'.format(loop_id))
                    graph.add_covisible_link(submap_node, loop_id)
                    last_node_inserted = loop_id
                else:
                    # No localization, add just traversability link
                    print('>> No localization found, adding traversability link...')
                    graph.create_regional_node(submap_node)
                    last_node_inserted = graph.last_inserted

        # 2. Plot graph
        gui.plot_topological_slam(graph, current_node=submap_node, frame=None, loop_id=loop_id, plot = plot_nodes, keyboard = False, wait = 300)

        # 3. Alternatively, save the update for later plotting
        graph.save_update(submap_idx)
                
    return graph

def topometric_slam_reuse(conf, net, submaps, image_size, output_dim, transform, 
                             graph_data=None, vg=False, plot=False, ms=[1], msp=1):
    """ Process topological nodes coming from an external SLAM algorithm and intermediate frames between them.
        Every node is added to the graph, or fused to an existing node if a loop closure is found.
        Intermediate frames are used to update the localization probability of the next node.
    """
    # Create likelihood estimator
    likelihood_estimator = LikelihoodEstimator(vg, net=net, threshold=conf['sim_threshold'], cov_threshold=conf['cov_threshold'], use_mlp=conf['use_mlp'])

    # Create graph and get conf values
    graph = TopologicalMapSLAM(conf, output_dim, likelihood_estimator=likelihood_estimator, window=conf['window_size'], vg=vg)
    gui = GUIPlotter(len(submaps), sequence=conf['sequence'], experiment_name=conf['experiment_name'])

    # Reload graph
    logging.info('Loading previous graph...')
    if conf['reload_graph']:
        logging.info('Reloading graph descriptors...')
        graph_data = reload_graph_descriptors(graph_data, net, image_size, output_dim, transform, vg, ms, msp)
    graph.initialize_graph_v2(graph_data, conf['initial_localization'])
    last_node_inserted = graph.last_inserted
     
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    
    loop_id = -1
    plot_frames = False
    plot_nodes = False
    debug = False

    # Mode: update or localize
    mode = conf['mode']

    last_localization = 0

    #### TODO: Number of nodes should be increased counting from the last inserted
    # Submap indexes should be somehow different to the ones in the graph
    # For now we just localize

    # Iterate through every node in nodes
    for submap_idx, node in enumerate(submaps):
        if submap_idx < conf['start']:
            print('>> Skipping node {}...'.format(submap_idx))
            continue

        print('>> Processing node {} of {}...'.format(submap_idx, len(submaps)))
        # 1. Now process the node itself
        node_loader = torch.utils.data.DataLoader(
                ImagesFromListVG(root='', images=node, imsize=image_size, transform=transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
        
        # Get descriptors for every image in the node
        with torch.no_grad():
            submap_node = None
            for img_idx, input in enumerate(node_loader):
                input = input.cuda()
                image_path = submaps[submap_idx][img_idx]

                # Extract descriptor
                descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)
                descriptor = descriptor.view(descriptor.shape[0], 1)

                if submap_node is None:
                    n_frame = os.path.basename(image_path).replace('.png', '').zfill(5)
                    submap_node = MultiNode(descriptor, image_path, n_frame, submap_idx, probability = 0.0)
                else:
                    submap_node.add_more_views(descriptor, image_path)

            if conf['filter']:
                submap_node = likelihood_estimator.check_intra_covisibility(submap_node, save=False)

            graph.localized = -1
            # Graph is already initialized, decide which link to insert
            localization_status, loop_id = graph.localize_submap_v2(submap_node, submap_node.descriptor)
            # Plot graph
            gui.plot_topological_slam(graph, current_node=submap_node, frame=None, loop_id=loop_id, plot = plot_nodes, keyboard = False, wait = 300)

            # localization_status = False
            if mode == 'slam':
                # Do SLAM with the previous map
                if localization_status:
                    # We have a localization, add covisibility link (and maybe traversability link)
                    print('>> Localization found in node {}!'.format(loop_id))
                    submap_node.recently_added = True # Added during the SLAM process
                    graph.localized = loop_id
                    graph.add_covisible_link(submap_node, loop_id, slam=True)
                    graph.update_position(loop_id)
                    last_localization = 0
                    graph.state = "ok"
                else:
                    # No localization, add just traversability link
                    print('>> No localization found, increase uncertainty......')
                    graph.increase_uncertainty()
                    last_localization += 1
                graph.localizations.append((submap_idx, loop_id))
                if last_localization > 4:
                    graph.state = "lost"
                    print(f'>> No localization found for {last_localization} nodes, system is lost...')
            else:
                # Just localize the node
                if localization_status:
                    # We have a localization, change the current position
                    print('>> Localization found in node {}!'.format(loop_id))
                    graph.update_position(loop_id)
                else:
                    # No localization, increase window size
                    print('>> No localization found, increase uncertainty...')
                    graph.increase_uncertainty()
                graph.localizations.append((submap_idx, loop_id))

        # 2. Plot graph
        gui.plot_topological_slam(graph, current_node=submap_node, frame=None, loop_id=loop_id, plot = plot_nodes, keyboard = False, wait = 300)

        # 3. Alternatively, save the update for later plotting
        graph.save_update_reuse(submap_idx)
                
    return graph

def reload_graph_descriptors(graph_data, net, image_size, output_dim, transform, vg=False, ms=[1], msp=1):
    likelihood_estimator = LikelihoodEstimator(vg)
    start_time = datetime.now()
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    new_graph_descriptors = []
    for id in graph_data['node_ids']:
        node_descriptors = []
        print(f"Reloading node {id}...")
        for node_images in graph_data['images'][id]:
            # creating dataset loader
            loader = torch.utils.data.DataLoader(
                    ImagesFromListVG(root='', images=node_images, imsize=image_size, transform=transform),
                    batch_size=1, shuffle=False, num_workers=8, pin_memory=True
                )
            
            # Initialize torch tensor for descriptors DxN
            descriptors = torch.zeros((output_dim, len(node_images)))
            for i, input in enumerate(loader):
                input = input.cuda()

                # Extract descriptor
                descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)
                descriptors[:,i] = descriptor

            node_descriptors.append(descriptors)
        
        new_graph_descriptors.append(node_descriptors)

    # for node_images in graph_data['images']:
    #     # creating dataset loader
    #     loader = torch.utils.data.DataLoader(
    #             ImagesFromListVG(root='', images=node_images, imsize=image_size, transform=transform),
    #             batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    #         )
        

    #     # Initialize torch tensor for descriptors DxN
    #     descriptors = torch.zeros((output_dim, len(node_images)))
    #     for i, input in enumerate(loader):
    #         input = input.cuda()

    #         # Extract descriptor
    #         descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)
    #         descriptors[:,i] = descriptor

    #     new_graph_descriptors.append(descriptors)

    graph_data['descriptors'] = new_graph_descriptors
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
        
    return graph_data
