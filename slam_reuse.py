import os
import sys
import torch
import vg_networks.parser_vg as parser
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
import torchvision.transforms as T
import pickle

import test
import vg_networks.util as util
import vg_networks.commons as commons
from vg_networks import network
from prettytable import PrettyTable
import numpy as np
import time
from topological.topometric import topometric_slam_reuse
from settings import DATA_PATH, EVAL_PATH

logging.getLogger('PIL').setLevel(logging.WARNING)

def get_configuration(conf):
    """Get the configuration from the name of the folder
    """
    # Split conf to get the parameters
    split_conf = conf.split("_")

    new_args = {}
    new_args["backbone"] = split_conf[0]
    new_args["aggregation"] = split_conf[1]

    if new_args["backbone"] == "cct384":
        new_args["resize"] = [384,384]
    elif new_args["backbone"] == "endofm":
        new_args["resize"] = [224,224]

    if new_args["aggregation"] == "netvlad":
        short_sim = 0.6
    else:
        short_sim = 0.92

    if split_conf[2] == "midl" or split_conf[2] == "colonmapper":
        new_args["l2"] = "after_pool"
        radenovic = True
        short_sim = 0.95
    else:
        new_args["l2"] = "before_pool"
        radenovic = False

    if new_args["aggregation"] == "salad":
        new_args["l2"] = "none"

    return new_args, short_sim, radenovic

def take_int(elem):
    # Get filename from path
    elem = os.path.basename(elem)
    return int(elem.replace('.png',''))

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################

# Extract configuration from the name of the folder. It is the parent folder of the model
model_conf = os.path.basename(os.path.dirname(args.resume))
new_args, short_sim, radenovic = get_configuration(model_conf)
args.__dict__.update(new_args)
print(args.resize)

model = network.GeoLocalizationNet(args)
model = model.to(args.device)

if args.aggregation in ["netvlad", "crn"]:
    args.features_dim *= args.netvlad_clusters

# Load the model from path
if radenovic:
    logging.info(f"Loading Radenovic model from {args.resume}")
    state = torch.load(args.resume)
    state_dict = state["state_dict"]
    model_keys = model.state_dict().keys()
    renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
    model.load_state_dict(renamed_state_dict)
else:
    logging.info(f"Resuming newest model from {args.resume}")
    model = util.resume_model(args, model)

# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)
model.eval()
model.cuda()

# Transformations
base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

######################################### DATASET #########################################


map_sequences = args.sequence_map
loc_sequences = args.sequence_loc
start_processing = args.start_processing

# Check if map_sequences and loc_sequences are lists
if not isinstance(map_sequences, list):
    map_sequences = [map_sequences]
    loc_sequences = [loc_sequences]
    start_processing = [int(x) for x in [start_processing]]


for i, sequence_to_load in enumerate(map_sequences):
    sequence_to_loc = loc_sequences[i]
    sequence_start = start_processing[i]
    
    print('>> Reading SLAM results from {}...'.format(DATA_PATH))
    print('>> Load graph from sequence: {}'.format(sequence_to_load))
    print('>> Localize sequence: {}'.format(sequence_to_loc))

    ### 1. First read submaps to localize ###
    # Load images
    submaps = []

    sequence_to_loc_path = os.path.join(DATA_PATH, sequence_to_loc)
    # Check if the sequence is a folder
    if not os.path.isdir(sequence_to_loc_path):
        assert False, 'Sequence {} does not exist.'.format(sequence_to_loc)

    print('>> Reading submaps for sequence {}...'.format(sequence_to_loc))
    submap_paths = sorted(os.listdir(os.path.join(DATA_PATH, sequence_to_loc)))
    for submap in submap_paths:
        submap_path = os.path.join(sequence_to_loc_path, submap)
        if os.path.isdir(submap_path):
            # 1. Get submap keyframes
            keyframes_path = os.path.join(submap_path, 'KeyFrames')
            # Get the list of files in the folder and sort them
            imgs = os.listdir(keyframes_path)
            sorted_imgs = sorted(imgs)
            sorted_imgs = [os.path.join(keyframes_path, img) for img in sorted_imgs]

            # Append the sorted list to the 'submaps' list
            submaps.append(sorted_imgs)

    ### 2. Load the previously built graph ###
    graph_name = '{}_withdrawal.pkl'.format(sequence_to_load)

    graph_pkl = join(DATA_PATH, 'reuse', graph_name)
    with open(graph_pkl, 'rb') as f:
        graph_dict = pickle.load(f)
    start = args.graph_start
    end = args.graph_end if args.graph_end != 0 else len(graph_dict['node_ids'])
    nodes_range = range(start, end)

    ids = []
    n_frames = []
    images = []
    descriptors = []
    edges = []
    for i in nodes_range:
        ids.append(graph_dict['node_ids'][i])
        n_frames.append(graph_dict['n_frames'][i])
        images.append(graph_dict['images'][i])
        descriptors.append(graph_dict['descriptors'][i])
        edges = graph_dict['edges']

    graph_data = {'node_ids': ids, 'n_frames': n_frames, 'images': images, 'descriptors': descriptors, 'edges': edges}


    ### 3. Configs and run ###
    print('>> Run topo-metric SLAM for sequence {}...'.format(sequence_to_loc))

    # Configs for local matching
    verifier_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'superpoint': {
                'detection_threshold': 0.00005,
                'max_num_keypoints': 2048
            },
            'lightglue': {
                'features': 'superpoint',
                'depth_confidence': -1,
                'width_confidence': -1,
                'filter_threshold': 0.1,
            }
        }
    
    if args.use_lightglue == False:
        verifier_config = None
    
    conf = {'sequence': sequence_to_loc,
            'radius_bayesian': 2,
            'nodes_similarity': 3,
            'multi_descriptor_node': False,
            'max_skipped': 7,
            'matching_node': 'last',
            'data_path': sequence_to_loc_path + '/',
            'graph_path': graph_name,
            'reload_graph': args.reload_graph,
            'initial_localization': args.initial_localization,
            'start': sequence_start,
            'short_sim': short_sim,
            'use_mlp': args.use_mlp,
            'only_LG': args.only_LG,
            'only_similarity': args.only_similarity,
            'miccai': args.miccai,
            'single_candidate': args.single_candidate,
            'sim_threshold': args.sim_threshold,
            'cov_threshold': args.cov_threshold,
            'matches_threshold': args.matches_threshold,
            'voting_threshold': args.voting_threshold,
            'filter': args.filter,
            'dynamic_window': args.dynamic_window,
            'experiment_name': args.experiment_name,
            'window_size': args.window_size,
            'use_intermediate': args.use_intermediate,
            'mode': args.mode,
            'verification': verifier_config,
            'slam_verification': args.slam_verification} #None

    start_time = datetime.now()
    start = time.time()
    graph = topometric_slam_reuse(conf, model, submaps, args.resize, args.features_dim, transform=base_transform, 
                                graph_data=graph_data, vg=True, plot=args.plot)
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
    end = time.time()
    runtime = (end - start)

    graph.save_reuse_results(sequence_to_loc, args.experiment_name)
    print('>> Results were saved.')

    graph.save_runtime(runtime, args.experiment_name, folder='reuse')
    print('>> Runtime were saved.')