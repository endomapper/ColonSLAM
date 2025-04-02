from lightglue.utils import load_image, rbd, mask_out
from os.path import join
import numpy as np
import os
import torch

def verify_node(extractor, matcher, map_node, query_submap, exhaustive=False):
    # Sample images from submap
    query_images = query_submap.image_paths
    max_matches = 0
    best_pair = None
    
    # Accelerate search
    if not exhaustive:
        if len(query_images) > 2:
            query_images = [query_images[0], query_images[len(query_images)//2], query_images[-1]]
    elif exhaustive == 'mid':
        # Extract 1/2 of the images, minumum 3
        if len(query_images) > 5:
            query_images = query_images[::len(query_images)//2]   

    for image0_name in query_images:
        image0, mask0 = load_image(image0_name, mask=True)

        # Extract features
        feats0 = extractor.extract(image0.cuda())
        feats0 = mask_out(feats0, mask0.cuda())

        for n, submap in enumerate(map_node.covisible_nodes):
            submap_images = submap.image_paths
            # Get three images
            if not exhaustive:
                if len(submap_images) > 2:
                    submap_images = [submap_images[0], submap_images[len(submap_images)//2], submap_images[-1]]
            elif exhaustive == 'mid':
                # Extract 1/2 of the images, minumum 3
                if len(submap_images) > 5:
                    submap_images = submap_images[::len(submap_images)//2]

            for image1_name in submap_images:
                image1, mask1 = load_image(image1_name, mask=True)
                feats1 = extractor.extract(image1.cuda())
                feats1 = mask_out(feats1, mask1.cuda())

                # match the features
                matches01 = matcher({'image0': feats0, 'image1': feats1})
                b_feats0, b_feats1, b_matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

                kpts0, kpts1, matches = b_feats0["keypoints"], b_feats1["keypoints"], b_matches01["matches"]
                m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

                if len(m_kpts0) > max_matches:
                    max_matches = len(m_kpts0)
                    best_pair = (image0_name, image1_name, m_kpts0, m_kpts1)

    return max_matches, best_pair
                    


