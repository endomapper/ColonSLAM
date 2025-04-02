
import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet18conv4",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5",
                                 "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                 "cct384", "vit", "endofm"], help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default="netvlad", choices=["netvlad", "gem", "spoc", "mac", "rmac", "crn", "rrm",
                                                                               "cls", "seqpool"])
    parser.add_argument('--netvlad_clusters', type=int, default=64, help="Number of clusters for NetVLAD layer.")
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")
    parser.add_argument("--off_the_shelf", type=str, default="imagenet", choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    parser.add_argument("--resnet_layer", type=str, default="layer3", choices=['layer1', 'layer2', 'layer3'])
    parser.add_argument("--aspect_ratio", type=str, default="resize", choices=['central_crop', 'resize'])
    parser.add_argument("--trainable_vit_blocks", type=int, default=0, help="Number of ViT blocks to fine-tune")
    parser.add_argument('--salad_clusters', type=int, default=64, help="Number of clusters for SALAD.")
    parser.add_argument('--cluster_dim', type=int, default=128, help="Dimension for SALAD clusters.")
    parser.add_argument('--token_dim', type=int, default=256, help="Dimension por global token projection.")
    parser.add_argument('--patch_size', type=int, default=14, help="Size of the patches used by ViT.")
    parser.add_argument('--use_mlp', action='store_true', help="Use MLP for descriptor comparison.")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[480, 640], nargs=2, help="Resizing shape for images (HxW).")

    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default='/media/jmorlana/105d508b-cd9c-4865-9335-c504d728aff1/HCULB', help="Path with all datasets")
    parser.add_argument("--sequences",  nargs='+', default=["027", "035", "036", "097", "098"], help="Sequences to test on")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")

    # ColonSLAM parameters
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment")
    parser.add_argument("--likelihood_method", type=str, default='topk',
                        help="Method for estimating likelihood")
    parser.add_argument("--localization_mode", type=str, default='nodes',
                        help="Wheter to localize using nodes or spreaded images")
    parser.add_argument('--plot', dest='plot', default=False, action='store_true',
                        help='Wheter to show or not the GUI')
    parser.add_argument("--voting_threshold", type=float, default=0.0,
                        help="Reduce likelihood if score is not above certain threshold")
    parser.add_argument('--filter', dest='filter', default=False, action='store_true',
                        help='Check intra covisibility or not')
    parser.add_argument('--miccai', dest='miccai', default=False, action='store_true',
                        help='Use MICCAI config or not')
    parser.add_argument('--single_candidate', dest='single_candidate', default=False, action='store_true',
                        help='Verify with only one candidate')
    parser.add_argument('--use_lightglue', action='store_true', help="Verify with LightGlue matching.")
    parser.add_argument('--only_LG', action='store_true', help="Estimate loops with only LightGlue matching.")
    parser.add_argument('--matches_threshold', type=int, default=100, help="Match threshold.")
    parser.add_argument('--only_similarity', action='store_true', help="Estimate loops with only network similarity.")
    parser.add_argument('--sim_threshold', type=float, default=0.95, help="Threshold for similarity.")
    parser.add_argument('--cov_threshold', type=float, default=0.95, help="Threshold for similarity.")
    parser.add_argument('--window_size', type=int, default=None, help="To use a local window or not.")
    parser.add_argument('--dynamic_window', action='store_true', help="Use dynamic window for localization.")
    parser.add_argument('--use_intermediate',  action='store_true', help="Use intermediate frames.")
    # Reload a previous graph
    parser.add_argument('--reload_graph', dest='reload_graph', default=False, action='store_true',
                        help='Wheter to load a previously built graph')
    parser.add_argument('--sequence_map', default=["027", "097"], help="Load the graph from this sequence")
    parser.add_argument('--sequence_loc', default=["035", "098"], help= "Localize this sequence")
    parser.add_argument('--graph_start', type=int, default=0,
                        help='Start node for loaded graph')
    parser.add_argument('--graph_end', type=int, default=0,
                        help='End node for loaded graph')
    parser.add_argument('--initial_localization', type=int, default=0,
                        help='Initial localization when using a loaded graph')
    parser.add_argument('--start_processing', default=[34, 31],
                        help='First submap to start SLAM processing')
    parser.add_argument('--mode', default="slam", help="Localize or update the previous graph")
    parser.add_argument('--slam_verification', dest='slam_verification', default=False, action='store_true',
                        help='Wheter to run an additional LF verification when doing SLAM') 
    args = parser.parse_args()
    
    if args.datasets_folder is None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    if args.aggregation == "crn" and args.resume is None:
        raise ValueError("CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None.")
    
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")
    
    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5", "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")
    
    if args.pca_dim is not None and args.pca_dataset_folder is None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    if args.backbone == "vit":
        if args.resize != [224, 224] and args.resize != [384, 384]:
            raise ValueError(f'Image size for ViT must be either 224 or 384 {args.resize}')
    if args.backbone == "cct384":
        if args.resize != [384, 384]:
            raise ValueError(f'Image size for CCT384 must be 384, but it is {args.resize}')
    
    if args.backbone in ["alexnet", "vgg16", "resnet18conv4", "resnet18conv5",
                         "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5"]:
        if args.aggregation in ["cls", "seqpool"]:
            raise ValueError(f"CNNs like {args.backbone} can't work with aggregation {args.aggregation}")
    if args.backbone in ["cct384"]:
        if args.aggregation in ["spoc", "mac", "rmac", "crn", "rrm"]:
            raise ValueError(f"CCT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls, seqpool]")
    if args.backbone == "vit":
        if args.aggregation not in ["cls", "gem", "netvlad"]:
            raise ValueError(f"ViT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls]")

    return args
