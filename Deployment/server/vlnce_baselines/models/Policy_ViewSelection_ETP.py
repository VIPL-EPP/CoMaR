from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from vlnce_baselines.models.etp.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses

from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.models.graph_utils import  MAX_DIST
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, length2mask, angle_feature_torch, pad_tensors, gen_seq_masks, get_angle_fts, get_angle_feature, get_point_angle_feature, calculate_vp_rel_pos_fts, calc_position_distance,pad_tensors_wgrad)
import math
from PIL import Image
import cv2
import open3d as o3d
from torch_kdtree import build_kd_tree
from vlnce_baselines.models.utils import  *
from vlnce_baselines.models.feature_fields import Feature_Fields
from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, angle_feature_torch, length2mask)

from vlnce_baselines.models.pret.PRET import PRET

class Net(nn.Module):
    def __init__(
        self, model_config,
    ):
        super().__init__()
        self.net = ETP(model_config)

    def forward(self):
        pass


class ETP(nn.Module):
    def __init__(
        self, model_config, 
    ):
        super().__init__()

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the ETP model ...')
        self.vln_bert = get_vlnbert_models(config=model_config)

        self.drop_env = nn.Dropout(p=0.4)

        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
        
        self.rgb_encoder = CLIPEncoder('/data/3dg-LF/ViT-L-14-336px.pt',self.device)
        # self.feature_fields = Feature_Fields(batch_size=1, device=self.device, mode='habitat',camera_intrinsic=None, bev=True) # Initialize the feature fields
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
        
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # 逆时针
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # 对应到逆时针
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

        batch_size = self.pano_angle_fts.shape[0]

        self.headings = [0 for i in range(batch_size)]
        self.positions = [0 for i in range(batch_size)]
        self.action_step = 0
        self.STOP_embedding = nn.Parameter(torch.zeros(1, 1, 768))
        self.pret = PRET(self.device)
        self.eval()

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return 1

    def preprocess_depth(self, depth):
        # depth - (B, H, W, 1) torch Tensor

        min_depth = 0.  # !!!!!!!!!!! This is the setting for R2R
        max_depth = 10. # !!!!!!!!!!! This is the setting for R2R

        # Column-wise post-processing
        depth = depth * 1.0
        H = depth.shape[1]
        depth_max, _ = depth.max(dim=1, keepdim=True)  # (B, H, W, 1)
        depth_max = depth_max.expand(-1, H, -1, -1)
        depth[depth == 0] = depth_max[depth == 0]

        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        depth = depth / 100.
        return depth


    def forward(self, mode=None, waypoint_predictor=None,
                txt_ids=None, txt_masks=None, txt_embeds=None, 
                batch_angle_idxes=None, batch_distance_idxes=None, observations=None, in_train=True,
                rgb_fts=None, dep_fts=None, loc_fts=None, 
                nav_types=None, view_lens=None,
                gmap_vp_ids=None, gmap_step_ids=None,
                gmap_img_fts=None, gmap_pos_fts=None, 
                gmap_masks=None, gmap_visited_masks=None, gmap_pair_dists=None, stepk=None, global_metric_num=None, predicted_bev_fts=None,
                path_features=None, path_padding_mask=None, local_features=None, local_padding_mask=None,
                path_feature=None, padding_mask=None, unvisited_or_current=None,
                gmap_features=None, gmap_logits=None, visited_mask=None):


        if mode == 'language':
            encoded_sentence = self.vln_bert.forward_txt(
                txt_ids, txt_masks,
            )
            return encoded_sentence
        

        elif mode == 'feature_field':
            # batch_size = observations['instruction'].size(0)
            batch_size = len(observations)
            observations = {
                'rgb': torch.cat([observations[b]['rgb'].unsqueeze(0) for b in range(batch_size)], 0),
                'depth': torch.cat([observations[b]['depth'].unsqueeze(0) for b in range(batch_size)], 0)
                }
            NUM_IMGS = 1

            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            
            # reverse the order of input images to clockwise
            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if 'depth' == k:  # You might need to double check the keys order
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                        depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                    a_count += 1

            obs_view = {}
            obs_view['depth'] = depth_batch
            obs_view['rgb'] = rgb_batch

            depth_height = self.vln_bert.feature_fields.args.input_height
            depth_width = self.vln_bert.feature_fields.args.input_width
            layer_width = self.vln_bert.feature_fields.args.mlp_net_width

            depth_batch_fts =  F.interpolate(obs_view['depth'], size=(depth_height,depth_width), mode='nearest').cpu().numpy()

            # depth_batch_fts = torch.zeros((obs_view['depth'].shape[0],depth_height,depth_width,1))
            # for i in range(obs_view['depth'].shape[0]):
            #     depth_batch_fts[i] = torch.tensor(cv2.resize(obs_view['depth'][i].cpu().numpy(), (depth_height, depth_width),  interpolation = cv2.INTER_NEAREST)).view(depth_height, depth_width,1)

            # depth_batch_fts = self.preprocess_depth(depth_batch_fts).view(batch_size,NUM_IMGS,depth_height*depth_width).numpy()

            with torch.no_grad():
                rgb_embedding, grid_batch_fts = self.rgb_encoder(obs_view)

            grid_batch_fts = grid_batch_fts.view(batch_size,NUM_IMGS,depth_height*depth_width,layer_width).cpu().numpy()

            self.vln_bert.feature_fields.update_feature_fields_habitat(self.positions, self.headings, depth_batch_fts, grid_batch_fts, num_of_views=NUM_IMGS)
            

        elif mode == 'semantic_map_for_waypoint':
            
            batch_size = len(observations)
            observations = {
                'rgb': torch.cat([observations[b]['rgb'].unsqueeze(0) for b in range(batch_size)], 0),
                'depth': torch.cat([observations[b]['depth'].unsqueeze(0) for b in range(batch_size)], 0)
                }
            # predicted_view_fts, _, _, _ = self.vln_bert.feature_fields.run_view_encode(self.positions,self.headings,visualization=False)
            predicted_panorama_fts = self.vln_bert.feature_fields.run_panorama_encode(self.positions,self.headings,visualization=False, panorama_encode=False)
            predicted_bev_fts = None
            #predicted_bev_fts, _, _ = self.vln_bert.feature_fields.run_bev_encode(self.positions,self.headings,visualization=False)

            rgb_embedding = predicted_panorama_fts

            ####################  Forward-facing CLIP features
            with torch.no_grad():
                clip_fts, _ = self.rgb_encoder(observations)

            for b in range(batch_size):
                rgb_embedding[b, 0] = clip_fts[b]

            ##########################

            # reverse the order of images back to counter-clockwise
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, 12, 768, 1, 1)
            
            rgb_feats = torch.cat((
                rgb_embed_reshape[:,0:1,:], 
                torch.flip(rgb_embed_reshape[:,1:,:], [1]),
            ), dim=1)
           
            
            rgb_feats = self.space_pool_rgb(rgb_feats)
                     

            ''' waypoint prediction ----------------------------- '''

            # for cand
            cand_rgb = []
            
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []
            for j in range(batch_size):

                angle_idxes = batch_angle_idxes[j]
                distance_idxes = batch_distance_idxes[j]

                # for angle & distance
                angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi       # 顺时针
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # 逆时针
                cand_angle_fts.append( angle_feature_torch(angle_rad_c) )
                cand_angles.append(angle_rad_cc.tolist())
                cand_distances.append( (distance_idxes*0.25).tolist() )
                # for img idxes
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # 逆时针
                img_idxes[img_idxes==12] = 0
                cand_img_idxes.append(img_idxes)
                # for rgb & depth
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                
            
            # for pano
            pano_rgb = rgb_feats                            # B x 12 x 2048
         
            pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
            pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

            # cand_angle_fts 顺时针
            # cand_angles 逆时针
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 2048]
             
                'cand_angle_fts': cand_angle_fts,   # [K x 4]
                'cand_img_idxes': cand_img_idxes,   # [K]
                'cand_angles': cand_angles,         # [K]
                'cand_distances': cand_distances,   # [K]

                'pano_rgb': pano_rgb,               # B x 12 x 2048
               
                'pano_angle_fts': pano_angle_fts,   # 12 x 4
                'pano_img_idxes': pano_img_idxes,   # 12 
                'predicted_bev_fts': predicted_bev_fts
            }
            return outputs
        
        
        elif mode == 'panorama':
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(
                rgb_fts, loc_fts, nav_types, view_lens,
            )
            return outs

        elif mode == 'navigation':
            outs = self.vln_bert.forward_navigation(
                txt_embeds, txt_masks, 
                gmap_vp_ids, gmap_step_ids,
                gmap_img_fts, gmap_pos_fts, 
                gmap_masks, gmap_visited_masks, gmap_pair_dists
            )
            return outs
        
        elif mode == 'MAM':
            stop_embedding = self.STOP_embedding.expand(path_features.shape[0], 1, -1)
            local_features = torch.cat([stop_embedding, local_features], dim=1)
            local_padding_mask = F.pad(local_padding_mask, (1, 0), value=False)

            # forward path score
            path_tokens, local_tokens = self.pret.forward_MAM(
                txt_embeds, txt_masks,
                path_features, path_padding_mask,
                local_features, local_padding_mask)
            
            return local_tokens
        
        elif mode == 'CCM':
            path_logits, gmap_logits, residual_logits, final_logits = self.pret.forward_CCM(path_feature, gmap_features, gmap_logits, txt_embeds, txt_masks, padding_mask, visited_mask)
            # fused_logits = fused_logits - fused_logits.max(dim=1, keepdim=True).values
            return path_logits, gmap_logits, residual_logits, final_logits

