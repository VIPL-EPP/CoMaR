import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn
import math
from torch_kdtree import build_kd_tree
import open3d as o3d
import random
from joblib import Parallel
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm

# Model Settings
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    # Novel view settings
    parser.add_argument("--near", type=float, default=0., 
                        help='near distance')
    parser.add_argument("--far", type=float, default=10., 
                        help='far distance')
    parser.add_argument("--view_hfov", type=float, default=90., 
                        help='hfov angle of rendered view')
    parser.add_argument("--view_vfov", type=float, default=90., 
                    help='vfov angle of rendered view')
    parser.add_argument("--view_height", type=int, default=12, 
                        help='height of the rendered view')
    parser.add_argument("--view_width", type=int, default=12, 
                        help='width of the rendered view')
    parser.add_argument("--input_hfov", type=float, default=90., 
                        help='hfov angle of input view')
    parser.add_argument("--input_vfov", type=float, default=90., 
                    help='vfov angle of input view')
    parser.add_argument("--input_height", type=int, default=24, 
                        help='height of the input view')
    parser.add_argument("--input_width", type=int, default=24, 
                        help='width of the input view')

    parser.add_argument("--occupancy_unit_size", type=float, default=0.1, 
                        help='size of the occupancy unit, only sample points in occupancy space for faster speed')
    parser.add_argument("--feature_fields_search_radius", type=float, default=0.5, 
                        help='search radius for near features')
    parser.add_argument("--feature_fields_search_num", type=int, default=4, 
                        help='The number of searched near features')
    parser.add_argument("--mlp_net_layers", type=int, default=8, 
                        help='layers in mlp network')
    parser.add_argument("--mlp_net_width", type=int, default=768, 
                        help='channels per layer in mlp network')
    parser.add_argument("--N_samples", type=int, default=501, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=16,
                        help='number of fine samples per ray')

    # BEV map settings
    parser.add_argument("--top", type=float, default=0., 
                        help='top distance')
    parser.add_argument("--down", type=float, default=-1.6, 
                        help='down distance')
    parser.add_argument("--localization_map_height", type=int, default=168, 
                        help='height of the localization map')
    parser.add_argument("--localization_map_width", type=int, default=168, 
                    help='width of the localization map')
    parser.add_argument("--grid_map_height", type=int, default=24, 
                        help='height of the grid feature map')
    parser.add_argument("--grid_map_width", type=int, default=24, 
                    help='width of the grid feature map')
    parser.add_argument("--bev_pixel_size", type=float, default=0.1, 
                    help='the size of each pixel for localization map')

    parser.add_argument("--bev_occupancy_unit_size", type=float, default=0.1, 
                        help='size of the occupancy unit, only sample points in occupancy space for faster speed')
    parser.add_argument("--bev_feature_fields_search_radius", type=float, default=0.4, 
                        help='search radius for near features')
    parser.add_argument("--bev_feature_fields_search_num", type=int, default=4, 
                        help='The number of searched near features')
    parser.add_argument("--bev_mlp_net_layers", type=int, default=4, 
                        help='layers in mlp network')
    parser.add_argument("--bev_mlp_net_width", type=int, default=768, 
                        help='channels per layer in mlp network')
    parser.add_argument("--bev_N_samples", type=int, default=17, 
                        help='number of coarse samples per ray')
    parser.add_argument("--bev_N_importance", type=int, default=4,
                        help='number of fine samples per ray')

    return parser


def project_depth_to_3d(depth, intrinsic, R, T, depth_scale, depth_trunc, input_height, input_width): # Don't define in the Feature_Fields Class, to avoid the memory copy of entire Feature_Fields Class at each thread when parallel computing
    depth[depth==0] = 1 # filter out the noise
    o3d_depth = o3d.geometry.Image(depth.numpy().astype(np.uint16))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d.camera.PinholeCameraIntrinsic(depth.shape[1],depth.shape[0],intrinsic[0][0],intrinsic[1][1],intrinsic[0][2],intrinsic[1][2]), depth_scale=depth_scale, depth_trunc=depth_trunc)
    points = np.asarray(pcd.points)
    points = torch.tensor(points).view(depth.shape[0],depth.shape[1],3).permute(2,0,1).unsqueeze(0) # num_of_samples x channels x height x width
    points = F.interpolate(points, size=(input_height,input_width), scale_factor=None, mode='nearest').squeeze(0).permute(1,2,0).view(-1,3).numpy()

    points_mask = points[:,2] > 0.002 # filter out the noise

    return (points, points_mask)


class Feature_Fields(nn.Module):
    def __init__(self, batch_size=1, device='cuda', mode='habitat',camera_intrinsic=None, bev=False):
        super(Feature_Fields, self).__init__()
        """
        Instantiate Feature Fields model.
        """
        self.device = device
        self.mode = mode
        self.bev = bev
        parser = config_parser()
        args, unknown = parser.parse_known_args()
        self.args = args

        self.thread_pool = Parallel(n_jobs=8,backend='threading') # Parallel computing with multiple CPUs, default is 8
        if self.mode == 'habitat':
            self.sampled_rays = self.get_rays_habitat()
        else:
            self.sampled_rays = self.get_rays(camera_intrinsic)

        self.tcnn_encoder = tcnn.Network(
            n_input_dims=args.mlp_net_width,
            n_output_dims=args.mlp_net_width+1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": args.mlp_net_width,
                "n_hidden_layers": args.mlp_net_layers//2,
            },
        )

        self.tcnn_decoder = tcnn.Network(
            n_input_dims=args.mlp_net_width,
            n_output_dims=args.mlp_net_width,
            network_config={
                "otype": "CutlassMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": args.mlp_net_width,
                "n_hidden_layers": args.mlp_net_layers - args.mlp_net_layers//2,
            },
        )

        width = args.mlp_net_width
        scale = width ** -0.5

        self.fcd_position_embedding = nn.Sequential(
            nn.Linear(6, width),
            BertLayerNorm(width, eps=1e-12)
        )
        self.fcd_aggregation = nn.Sequential(
            nn.Linear(args.mlp_net_width*args.feature_fields_search_num, width),
            BertLayerNorm(width, eps=1e-12)
        )

        self.class_embedding = nn.Parameter(scale * torch.randn(1,width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(args.view_height*args.view_width + 1, width))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=width//64, dim_feedforward=4*width, dropout=0.1,
                 activation="gelu", batch_first=True
        )
        self.view_encoder = nn.TransformerEncoder(enc_layer, num_layers=4, norm=BertLayerNorm(width, eps=1e-12))

        self.panorama_positional_embedding = nn.Parameter(scale * torch.randn(12, width))
        self.panorama_encoder = nn.TransformerEncoder(enc_layer, num_layers=4, norm=BertLayerNorm(width, eps=1e-12))

        if self.bev == True:

            if self.mode == 'habitat':
                self.bev_sampled_rays = self.get_bev_rays_habitat()
            else:
                self.bev_sampled_rays = self.get_bev_rays()

            self.bev_tcnn_encoder = tcnn.Network(
                n_input_dims=args.bev_mlp_net_width,
                n_output_dims=args.bev_mlp_net_width+1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "None",
                    "n_neurons": args.bev_mlp_net_width,
                    "n_hidden_layers": args.bev_mlp_net_layers//2,
                },
            )

            self.bev_tcnn_decoder = tcnn.Network(
                n_input_dims=args.bev_mlp_net_width,
                n_output_dims=args.bev_mlp_net_width,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "None",
                    "n_neurons": args.bev_mlp_net_width,
                    "n_hidden_layers": args.bev_mlp_net_layers - args.bev_mlp_net_layers//2,
                },
            )
            self.bev_fcd_position_embedding = nn.Sequential(
                nn.Linear(6, width),
                BertLayerNorm(width, eps=1e-12)
            )
            self.bev_fcd_aggregation = nn.Sequential(
                nn.Linear(args.bev_mlp_net_width*args.bev_feature_fields_search_num, width),
                BertLayerNorm(width, eps=1e-12)
            )

            kernel_size = self.args.localization_map_height // self.args.grid_map_height
            self.bev_enc_conv = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=kernel_size, stride=kernel_size, bias=False)
            self.bev_positional_embedding = nn.Parameter(scale * torch.randn(args.grid_map_height*args.grid_map_width, width))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=width//64, dim_feedforward=4*width, dropout=0.1,
                     activation="gelu", batch_first=True
            )
            self.bev_encoder = nn.TransformerEncoder(enc_layer, num_layers=4, norm=BertLayerNorm(width, eps=1e-12))


        self.batch_size = batch_size
        self.global_fts = [[] for i in range(self.batch_size)]
        self.global_position_x = [[] for i in range(self.batch_size)]
        self.global_position_y = [[] for i in range(self.batch_size)]
        self.global_position_z = [[] for i in range(self.batch_size)]
        self.global_patch_scales = [[] for i in range(self.batch_size)]
        self.global_patch_directions = [[] for i in range(self.batch_size)]
        self.fcd = [[] for i in range(self.batch_size)]
        self.fcd_tree = [[] for i in range(self.batch_size)]
        self.occupancy_fcd_tree = [[] for i in range(self.batch_size)]

    def reset(self, batch_size=1, mode='habitat',camera_intrinsic=None, batch_gt_pcd_xyz=None, batch_gt_pcd_label=None):
        self.batch_size = batch_size
        self.mode = mode
        self.global_fts = [[] for i in range(self.batch_size)]
        self.global_position_x = [[] for i in range(self.batch_size)]
        self.global_position_y = [[] for i in range(self.batch_size)]
        self.global_position_z = [[] for i in range(self.batch_size)]
        self.global_patch_scales = [[] for i in range(self.batch_size)]
        self.global_patch_directions = [[] for i in range(self.batch_size)]
        self.fcd = [[] for i in range(self.batch_size)]
        self.fcd_tree = [[] for i in range(self.batch_size)]
        self.occupancy_fcd_tree = [[] for i in range(self.batch_size)]

        if self.mode == 'habitat':
            self.sampled_rays = self.get_rays_habitat()
        else:
            self.sampled_rays = self.get_rays(camera_intrinsic)

        if self.bev == True:
            if self.mode == 'habitat':
                self.bev_sampled_rays = self.get_bev_rays_habitat()
            else:
                self.bev_sampled_rays = self.get_bev_rays()

        if batch_gt_pcd_xyz != None: # Load the point cloud and instance label of 3d scenes
            self.gt_pcd_xyz = []
            self.gt_pcd_tree = []
            self.gt_pcd_label = []
            for i in range(self.batch_size):
                if batch_gt_pcd_xyz[i] != None:
                    gt_pcd_xyz = batch_gt_pcd_xyz[i].clone()
                    if mode == "habitat":
                        gt_pcd_xyz[:,0], gt_pcd_xyz[:,1], gt_pcd_xyz[:,2] =  batch_gt_pcd_xyz[i][:,0], batch_gt_pcd_xyz[i][:,1], batch_gt_pcd_xyz[i][:,2]-1.5 # Agent's height is 1.5 meters
                    elif mode == "scannet":
                        gt_pcd_xyz[:,0], gt_pcd_xyz[:,1], gt_pcd_xyz[:,2] = batch_gt_pcd_xyz[i][:,0], batch_gt_pcd_xyz[i][:,1], batch_gt_pcd_xyz[i][:,2]
                    elif mode == "structured3d":
                        gt_pcd_xyz[:,0], gt_pcd_xyz[:,1], gt_pcd_xyz[:,2] = batch_gt_pcd_xyz[i][:,0], batch_gt_pcd_xyz[i][:,1], batch_gt_pcd_xyz[i][:,2]

                    self.gt_pcd_xyz.append(gt_pcd_xyz) 
                    self.gt_pcd_tree.append(build_kd_tree(gt_pcd_xyz.to(torch.float32).to(self.device)))
                    self.gt_pcd_label.append(batch_gt_pcd_label[i].to(torch.int64).to(self.device)) 
                else:
                    self.gt_pcd_xyz.append(None)
                    self.gt_pcd_tree.append(None)
                    self.gt_pcd_label.append(None)

            del batch_gt_pcd_xyz, batch_gt_pcd_label

        else:
            self.gt_pcd_xyz = None
            self.gt_pcd_tree = None
            self.gt_pcd_label = None

    def pop(self,index):
        self.batch_size -= 1
        self.global_fts.pop(index)
        self.global_position_x.pop(index)
        self.global_position_y.pop(index)
        self.global_position_z.pop(index)
        self.global_patch_scales.pop(index)
        self.global_patch_directions.pop(index)
        self.fcd.pop(index)
        self.fcd_tree.pop(index)
        self.occupancy_fcd_tree.pop(index)
        if self.gt_pcd_xyz != None:
            self.gt_pcd_xyz.pop(index)
            self.gt_pcd_tree.pop(index)
            self.gt_pcd_label.pop(index)


    def initialize_camera_setting(self, hfov, vfov):
        self.args.input_hfov = hfov
        self.args.input_vfov = vfov

    def initialize_novel_view_setting(self, hfov, vfov):
        self.args.view_hfov = hfov
        self.args.view_vfov = vfov


    def delete_feature_fields(self): # Free the memory
        del self.global_fts, self.global_position_x, self.global_position_y, self.global_position_z, self.global_patch_scales, self.global_patch_directions,self.fcd, self.fcd_tree, self.occupancy_fcd_tree
        if self.gt_pcd_xyz != None:
            del self.gt_pcd_xyz, self.gt_pcd_tree, self.gt_pcd_label



    def get_rays_habitat(self):
        H = self.args.view_height
        W = self.args.view_width
        rel_y = np.expand_dims(np.linspace(self.args.near, self.args.far, self.args.N_samples),axis=0).repeat(H*W,axis=0)    
        hfov_angle = np.deg2rad(self.args.view_hfov)
        vfov_angle = np.deg2rad(self.args.view_vfov)
        half_H = H//2
        half_W = W//2
        tan_xy = np.array(([[i/half_W+1/W] for i in range(-half_W,half_W)])*H,np.float32) * math.tan(hfov_angle/2.)
        rel_direction = - np.arctan(tan_xy)
        rel_x = rel_y * tan_xy
        rel_z = rel_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,1)) * math.tan(vfov_angle/2.))
        rel_position = (rel_x,rel_y,rel_z)
        rel_dist = rel_y
        return (rel_position, rel_direction, rel_dist)


    def get_bev_rays_habitat(self):
        H = self.args.localization_map_height
        W = self.args.localization_map_width
        rel_z = np.expand_dims(np.linspace(self.args.top, self.args.down, self.args.bev_N_samples),axis=0).repeat(H*W,axis=0)    
        half_H = H//2
        half_W = W//2
        rel_x = np.array(([[i+0.5]*self.args.bev_N_samples for i in range(-half_W,half_W)])*H,np.float32) * self.args.bev_pixel_size
        rel_y = np.array([[[i-0.5]*self.args.bev_N_samples]*W for i in range(half_H,-half_H,-1)],np.float32).reshape((-1,self.args.bev_N_samples)) * self.args.bev_pixel_size
        rel_position = (rel_x,rel_y,rel_z)
        rel_direction = np.zeros(rel_x[...,-1:].shape)
        rel_dist = - rel_z
        return (rel_position, rel_direction, rel_dist)


    def get_rays(self,camera_intrinsic):
        N_spacing = (self.args.far - self.args.near) / self.args.N_samples
        sampled_points = o3d.geometry.PointCloud()
        for N_index in range(self.args.N_samples):
            N_distance = self.args.near + N_spacing * (N_index+1)
            N_depth = np.full((self.args.view_height,self.args.view_width),N_distance,dtype=np.float32)
            N_depth = o3d.geometry.Image(N_depth)
            N_points = o3d.geometry.PointCloud.create_from_depth_image(N_depth, o3d.camera.PinholeCameraIntrinsic(self.args.view_width,self.args.view_height,camera_intrinsic[0][0],camera_intrinsic[1][1],self.args.view_width/2,self.args.view_height/2), depth_scale=1., depth_trunc=1.)
            sampled_points += N_points
        sampled_points = np.asarray(sampled_points.points)

        rel_position = sampled_points.reshape((self.args.N_samples, self.args.view_height*self.args.view_width,3)).transpose((1,0,2))

        rel_direction = - np.arctan(rel_position[...,-1:,0]/rel_position[...,-1:,2])
        rel_dist = rel_position[...,2]
        return (rel_position, rel_direction, rel_dist)


    def get_bev_rays(self):
        H = self.args.localization_map_height
        W = self.args.localization_map_width
        rel_z = np.expand_dims(np.linspace(self.args.top, self.args.down, self.args.bev_N_samples),axis=0).repeat(H*W,axis=0)    
        half_H = H//2
        half_W = W//2
        rel_x = np.array(([[i+0.5]*self.args.bev_N_samples for i in range(-half_W,half_W)])*H,np.float32) * self.args.bev_pixel_size
        rel_y = np.array([[[i-0.5]*self.args.bev_N_samples]*W for i in range(half_H,-half_H,-1)],np.float32).reshape((-1,self.args.bev_N_samples)) * self.args.bev_pixel_size
        rel_position = np.concatenate((np.expand_dims(rel_x,-1),np.expand_dims(-rel_z,-1),np.expand_dims(rel_y,-1)),axis=-1).reshape(-1,3)
        rel_direction = np.zeros(rel_x[...,-1:].shape)
        rel_dist = rel_position[...,1]
        return (rel_position, rel_direction, rel_dist)


    def run_region_encode(self, sample_ft_neighbor_embedding, sample_ft_neighbor_xyzds):

        sample_ft_neighbor_embedding = sample_ft_neighbor_embedding.view(-1,self.args.mlp_net_width*self.args.feature_fields_search_num).to(torch.float16)

        sample_ft_neighbor_xyzds = self.fcd_position_embedding(sample_ft_neighbor_xyzds).view(-1,self.args.mlp_net_width*self.args.feature_fields_search_num).to(torch.float16)

        sample_input = self.fcd_aggregation(sample_ft_neighbor_embedding+sample_ft_neighbor_xyzds)
        encoded_input = self.tcnn_encoder(sample_input)
        encoded_input, density = encoded_input[::,:-1], encoded_input[::,-1]

        encoded_input = encoded_input + sample_input # Residual
        outputs = (self.tcnn_decoder(encoded_input) + encoded_input).view(-1,self.args.N_importance, self.args.mlp_net_width) # Residual
        density = density.view(-1,self.args.N_importance)

        return outputs.to(torch.float16), density.to(torch.float16)   


    def run_bev_region_encode(self, sample_ft_neighbor_embedding, sample_ft_neighbor_xyzds):

        sample_ft_neighbor_embedding = sample_ft_neighbor_embedding.view(-1,self.args.bev_mlp_net_width*self.args.bev_feature_fields_search_num).to(torch.float16)

        sample_ft_neighbor_xyzds = self.bev_fcd_position_embedding(sample_ft_neighbor_xyzds).view(-1,self.args.bev_mlp_net_width*self.args.bev_feature_fields_search_num).to(torch.float16)

        sample_input = self.bev_fcd_aggregation(sample_ft_neighbor_embedding+sample_ft_neighbor_xyzds)

        encoded_input = self.bev_tcnn_encoder(sample_input)
        encoded_input, density = encoded_input[::,:-1], encoded_input[::,-1]

        encoded_input = encoded_input + sample_input # Residual
        outputs = (self.bev_tcnn_decoder(encoded_input) + encoded_input).view(-1,self.args.bev_mlp_net_width) # Residual

        return outputs.to(torch.float16), density.to(torch.float16)  


    def raw2feature(self, sample_feature, sample_density, rel_dist, topk_inds):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            feature: [num_rays, num_important_samples along ray, dimension of feature]. Prediction from model.
            density: [num_rays, num_important_samples along ray]. Prediction from model.
            rel_dist: [num_rays, num_samples along ray]. Integration time.
            topk_inds: [num_rays, num_important_samples along ray]. Important sample_id along ray
        Returns:
            feature_map: [num_rays, 768]. Estimated semantic feature of a ray.
            depth_map: [num_rays]. Estimated distance to camera.
        """
        sample_density = F.softplus(sample_density) # Make sure sample_density > 0.

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
        dists = torch.abs(rel_dist[...,1:] - rel_dist[...,:-1])
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)  # [N_rays, N_samples]
        density = torch.zeros(rel_dist.shape,dtype=sample_density.dtype,device=sample_density.device)     
        density = torch.scatter(density,1,topk_inds,sample_density)

        alpha = raw2alpha(density, dists)  # [N_rays, N_samples]
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        sample_weights = torch.gather(weights,1,topk_inds)

        feature_map = torch.sum(sample_weights[...,None] * sample_feature, -2)  # [N_rays, 768]
        feature_map = feature_map / torch.max(torch.linalg.norm(feature_map, dim=-1, keepdim=True),torch.tensor(1e-7,dtype=feature_map.dtype,device=feature_map.device))

        depth_map = torch.sum(weights * rel_dist, -1) / torch.max(torch.sum(weights, -1),torch.tensor(1e-7,dtype=weights.dtype,device=weights.device))

        return feature_map, depth_map


    def bev_raw2feature(self, sample_feature, sample_density):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            sample_feature: [num_rays, num_important_samples along ray, dimension of feature]. Prediction from model.
            sample_density: [num_rays, num_important_samples along ray]. Prediction from model.
        Returns:
            feature_map: [num_rays, 768]. Estimated semantic feature of a ray.
        """
        sample_weights = F.softmax(sample_density,dim=-1)
        feature_map = torch.sum(sample_weights[...,None] * sample_feature, -2)  # [N_rays, 768]
        feature_map = feature_map / torch.max(torch.linalg.norm(feature_map, dim=-1, keepdim=True),torch.tensor(1e-7,dtype=feature_map.dtype,device=feature_map.device))
        return feature_map


    def run_view_encode(self, batch_position=None, batch_direction=None, batch_rot=None, batch_trans=None, visualization=False):

        batch_feature_map = []
        batch_gt_label = []
        with torch.no_grad():
            for batch_id in range(self.batch_size):
                if self.mode == 'habitat':
                    position = batch_position[batch_id].copy()
                    position[0], position[1], position[2] = batch_position[batch_id][0], - batch_position[batch_id][2], batch_position[batch_id][1] # Note to swap y,z axis, and -y
                    camera_direction = batch_direction[batch_id]
                else:
                    R = batch_rot[batch_id]
                    T = batch_trans[batch_id]

                    points = np.array([[0.,0.,0.]])
                    points = (R @ points.T + T).T
                    position = points[0] # Get position of camera

                    points = np.array([[0.,0.,1.]])
                    points = (R @ points.T + T).T

                    camera_direction = self.get_heading_angle(points)[0] # Get direction of camera

                camera_x, camera_y, camera_z = position[0], position[1], position[2]
                scene_fts, patch_directions, patch_scales, fcd, fcd_tree, occupancy_fcd_tree = self.global_fts[batch_id], self.global_patch_directions[batch_id], self.global_patch_scales[batch_id], self.fcd[batch_id], self.fcd_tree[batch_id], self.occupancy_fcd_tree[batch_id]

                patch_directions = torch.tensor(patch_directions - camera_direction, dtype=torch.float32,device=self.device)

                patch_scales = torch.tensor(patch_scales, dtype=torch.float32, device=self.device).unsqueeze(-1)

                fcd_points = fcd.to(self.device)
                if self.mode == 'habitat':
                    rel_position, rel_direction, rel_dist = self.sampled_rays
                    rel_x, rel_y, rel_z = rel_position
                    ray_x = rel_x * math.cos(camera_direction) - rel_y * math.sin(camera_direction) + camera_x
                    ray_y = rel_x * math.sin(camera_direction) + rel_y * math.cos(camera_direction) + camera_y

                    ray_z = rel_z + camera_z
                    ray_xyz = torch.tensor(np.concatenate((np.expand_dims(ray_x,-1),np.expand_dims(ray_y,-1),np.expand_dims(ray_z,-1)),axis=-1),dtype=torch.float32, device=self.device)
                else:
                    rel_position, rel_direction, rel_dist = self.sampled_rays
                    ray_xyz = (R @ rel_position.reshape((-1,3)).T + T).T

                    ray_xyz = torch.tensor(ray_xyz,dtype=torch.float32, device=self.device).view(self.args.view_height*self.args.view_width,self.args.N_samples,3)

                with torch.no_grad():
                    occupancy_query = ray_xyz.view(-1,3)
                    searched_occupancy_dists, searched_occupancy_inds = occupancy_fcd_tree.query(occupancy_query, nr_nns_searches=1) #Note that the cupy_kdtree distances are squared
                    occupancy_masks = (searched_occupancy_dists < self.args.feature_fields_search_radius).view(-1,)
                    occupancy_ray_xyz = ray_xyz.view(-1,3)[occupancy_masks]

                    occupancy_ray_k_neighbor_dists, occupancy_ray_k_neighbor_inds = fcd_tree.query(occupancy_ray_xyz, nr_nns_searches=self.args.feature_fields_search_num)

                searched_ray_k_neighbor_dists = torch.full((ray_xyz.shape[0]*ray_xyz.shape[1],self.args.feature_fields_search_num),self.args.feature_fields_search_radius,dtype=torch.float32, device=self.device)
                searched_ray_k_neighbor_dists[occupancy_masks] = occupancy_ray_k_neighbor_dists

                searched_ray_k_neighbor_inds = torch.full((ray_xyz.shape[0]*ray_xyz.shape[1],self.args.feature_fields_search_num),-1,dtype=torch.int64, device=self.device)
                searched_ray_k_neighbor_inds[occupancy_masks] = occupancy_ray_k_neighbor_inds

                searched_ray_k_neighbor_dists = torch.sqrt(searched_ray_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
                searched_ray_k_neighbor_inds[searched_ray_k_neighbor_dists >= self.args.feature_fields_search_radius] = -1
                searched_ray_k_neighbor_dists[searched_ray_k_neighbor_dists >= self.args.feature_fields_search_radius] = self.args.feature_fields_search_radius


                searched_ray_k_neighbor_inds = searched_ray_k_neighbor_inds.view(ray_xyz.shape[0],ray_xyz.shape[1],self.args.feature_fields_search_num)
                searched_ray_k_neighbor_dists = searched_ray_k_neighbor_dists.view(ray_xyz.shape[0],ray_xyz.shape[1],self.args.feature_fields_search_num)

                tmp_distance = searched_ray_k_neighbor_dists.sum(-1)
                tmp_density = (1/tmp_distance)                
                topk_output = torch.topk(tmp_density, k=self.args.N_importance, dim=-1, largest=True)[1]
                topk_inds = torch.sort(topk_output, dim=-1)[0] # Search important sampled points

                sample_ray_xyz = torch.gather(ray_xyz,1,topk_inds.unsqueeze(-1).repeat(1,1,3))
                sample_ray_direction = rel_direction[::,-1]

                if self.gt_pcd_tree != None and self.gt_pcd_tree[batch_id] != None:
                    gt_ray_xyz = torch.gather(ray_xyz,1,topk_output.unsqueeze(-1).repeat(1,1,3))[...,0,:]
                    with torch.no_grad():
                        gt_dists, gt_inds = self.gt_pcd_tree[batch_id].query(gt_ray_xyz,nr_nns_searches=1)
                    gt_dists = torch.sqrt(gt_dists) #Note that the cupy_kdtree distances are squared
                    gt_label = self.gt_pcd_label[batch_id][gt_inds]
                    gt_label[gt_dists >= self.args.feature_fields_search_radius] = -100
                    gt_label[searched_ray_k_neighbor_inds.sum(-1).sum(-1) == -self.args.N_samples*self.args.feature_fields_search_num] = -100
                    batch_gt_label.append(gt_label)

                elif self.gt_pcd_tree != None and self.gt_pcd_tree[batch_id] == None:
                    batch_gt_label.append(None)

                if visualization: # Visualize the feature points of feature fields and sampled points of rays, set `True` for debug
                    rays_pcd=o3d.geometry.PointCloud()
                    rays_pcd.points = o3d.utility.Vector3dVector(ray_xyz.view(-1,3).cpu().numpy()) # ray_xyz, sample_ray_xyz
                    feature_pcd=o3d.geometry.PointCloud()
                    feature_pcd.points = o3d.utility.Vector3dVector( np.concatenate([self.global_position_x[batch_id].reshape((-1,1)),self.global_position_y[batch_id].reshape((-1,1)),self.global_position_z[batch_id].reshape((-1,1))],axis=-1) )
                    #gt_pcd=o3d.geometry.PointCloud()
                    #gt_pcd.points = o3d.utility.Vector3dVector(self.gt_pcd_xyz[batch_id])
                    #feature_pcd += gt_pcd
                    feature_pcd += rays_pcd
                    o3d.visualization.draw_geometries([feature_pcd])

                with torch.no_grad():
                    sample_feature_k_neighbor_dists, sample_feature_k_neighbor_inds = fcd_tree.query(sample_ray_xyz.view(-1,3), nr_nns_searches=self.args.feature_fields_search_num)

                sample_feature_k_neighbor_dists = torch.sqrt(sample_feature_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
                sample_feature_k_neighbor_inds[sample_feature_k_neighbor_dists >= self.args.feature_fields_search_radius] = -1
                sample_feature_k_neighbor_inds = sample_feature_k_neighbor_inds.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],self.args.feature_fields_search_num)


                sample_ft_neighbor_xyzds = torch.zeros((sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],self.args.feature_fields_search_num,6),dtype=torch.float32, device=self.device)
                idx = sample_feature_k_neighbor_inds 
                sample_ft_neighbor_xyzds[...,:3] = fcd_points[idx] - sample_ray_xyz.unsqueeze(-2)
                # Get the relative positions of features to the sampled point

                sample_ft_neighbor_x = sample_ft_neighbor_xyzds[...,0]
                sample_ft_neighbor_y = sample_ft_neighbor_xyzds[...,1]
                sample_ft_neighbor_xyzds[...,0] = sample_ft_neighbor_x * math.cos(-camera_direction) - sample_ft_neighbor_y * math.sin(-camera_direction)
                sample_ft_neighbor_xyzds[...,1] = sample_ft_neighbor_x * math.sin(-camera_direction) + sample_ft_neighbor_y * math.cos(-camera_direction)


                sample_ft_neighbor_xyzds[...,:3][idx==-1] = self.args.far

                sample_ray_direction = torch.tensor(sample_ray_direction,device=sample_ft_neighbor_xyzds.device)
                sample_ray_direction =  patch_directions[idx] - sample_ray_direction.unsqueeze(-1).unsqueeze(-1)
                sample_ray_direction = torch.cat((torch.sin(sample_ray_direction).unsqueeze(-1), torch.cos(sample_ray_direction).unsqueeze(-1)), dim=-1)

                sample_ft_neighbor_xyzds[...,3:5] = sample_ray_direction
                sample_ft_neighbor_xyzds[...,3:5][idx==-1] = 0
                sample_ft_neighbor_xyzds[...,5:] = patch_scales[idx]
                sample_ft_neighbor_xyzds[...,5:][idx==-1] = 0

                sample_ft_neighbor_embedding = scene_fts[idx.cpu().numpy()]
                sample_ft_neighbor_embedding = torch.tensor(sample_ft_neighbor_embedding,dtype=torch.float16,device=self.device)
                sample_ft_neighbor_embedding[idx==-1] = 0

                sample_feature, sample_density = self.run_region_encode(sample_ft_neighbor_embedding, sample_ft_neighbor_xyzds)
                rel_dist = torch.tensor(rel_dist,dtype=torch.float16,device=self.device)
                feature_map, depth_map = self.raw2feature(sample_feature, sample_density, rel_dist.view(-1,self.args.N_samples), topk_inds)
                feature_map = torch.cat((self.class_embedding,feature_map),dim=0)
                batch_feature_map.append(feature_map.unsqueeze(0))

            batch_feature_map = torch.cat(batch_feature_map,dim=0)
            batch_predicted_fts = self.view_encoder(batch_feature_map + self.positional_embedding)
            batch_view_fts, batch_region_fts = batch_predicted_fts[:,0], batch_predicted_fts[:,1:]
        return batch_view_fts, batch_region_fts, batch_feature_map[:,1:], batch_gt_label



    def run_panorama_encode(self, batch_position=None, batch_direction=None, batch_rot=None, batch_trans=None, visualization=False, panorama_encode=True):

        view_num = 12
        batch_panorama_fts = []
        with torch.no_grad():
            for view_id in range(view_num):
                batch_view_heading_angle = []
                batch_view_position = []
                batch_view_rot = []
                batch_view_trans = []
                for batch_id in range(self.batch_size):
                    if self.mode == 'habitat':
                        position = batch_position[batch_id]
                        camera_direction = batch_direction[batch_id]
                        batch_view_position.append(position)
                    
                        view_heading_angle = (view_id*(-math.pi/6) + camera_direction) % (2.*math.pi)
                        batch_view_heading_angle.append(view_heading_angle)
                    else:
                        R = batch_rot[batch_id]
                        T = batch_trans[batch_id]
                        batch_view_trans.append(T)
                    
                        pano_angle = view_id*(-math.pi/6)
                        pano_R = np.array([[math.cos(pano_angle),-math.sin(pano_angle),0.],
                                            [math.sin(pano_angle),math.cos(pano_angle),0.],
                                            [0.,0.,1.]]) # Rotation for panorama
                        R = pano_R @ R
                        batch_view_rot.append(R)
                

                with torch.no_grad(): # No grad for saving GPU memory
                    if self.mode == 'habitat':
                        batch_view_fts, batch_region_fts, batch_feature_map, batch_gt_label = self.run_view_encode(batch_position=batch_view_position, batch_direction=batch_view_heading_angle, visualization=visualization)
                        batch_panorama_fts.append(batch_view_fts)
                    else:
                        batch_view_fts, batch_region_fts, batch_feature_map, batch_gt_label = self.run_view_encode(batch_rot=batch_view_rot, batch_trans=batch_view_trans, visualization=visualization)
                        batch_panorama_fts.append(batch_view_fts)


            batch_panorama_fts = torch.cat(batch_panorama_fts,dim=0).view(view_num,self.batch_size,-1).permute(1,0,2) # batch_size, view_num, ft_dim
        
        if panorama_encode:
            batch_panorama_fts = batch_panorama_fts + self.panorama_positional_embedding
            batch_panorama_fts = self.panorama_encoder(batch_panorama_fts)
        
        return batch_panorama_fts


    def run_bev_encode(self, batch_position=None, batch_direction=None, batch_rot=None, batch_trans=None, visualization=False):

        batch_bev_feature_map = []
        batch_gt_label = []
        with torch.no_grad():
            for batch_id in range(self.batch_size):
                if self.mode == 'habitat':
                    position = batch_position[batch_id].copy()
                    position[0], position[1],position[2] = batch_position[batch_id][0], - batch_position[batch_id][2], batch_position[batch_id][1] # Note to swap y,z axis
                    camera_direction = batch_direction[batch_id]
                else:
                    R = batch_rot[batch_id]
                    T = batch_trans[batch_id]

                    points = np.array([[0.,0.,0.]])
                    points = (R @ points.T + T).T
                    position = points[0] # Get position of camera

                    points = np.array([[0.,0.,1.]])
                    points = (R @ points.T + T).T

                    camera_direction = self.get_heading_angle(points)[0] # Get direction of camera

                camera_x, camera_y, camera_z = position[0], position[1], position[2]
                scene_fts, patch_directions, patch_scales, fcd, fcd_tree, occupancy_fcd_tree = self.global_fts[batch_id], self.global_patch_directions[batch_id], self.global_patch_scales[batch_id], self.fcd[batch_id], self.fcd_tree[batch_id], self.occupancy_fcd_tree[batch_id]

                patch_directions = torch.tensor(patch_directions, dtype=torch.float32, device=self.device) - camera_direction

                patch_scales = torch.tensor(patch_scales, dtype=torch.float32).to(self.device).unsqueeze(-1)

                fcd_points = fcd.to(self.device)
                if self.mode == 'habitat':
                    rel_position, rel_direction, rel_dist = self.bev_sampled_rays
                    rel_x, rel_y, rel_z = rel_position
                    ray_x = rel_x * math.cos(camera_direction) - rel_y * math.sin(camera_direction) + camera_x
                    ray_y = rel_x * math.sin(camera_direction) + rel_y * math.cos(camera_direction) + camera_y
                    ray_z = rel_z + camera_z
                    ray_xyz = torch.tensor(np.concatenate((np.expand_dims(ray_x,-1),np.expand_dims(ray_y,-1),np.expand_dims(ray_z,-1)),axis=-1),dtype=torch.float32, device=self.device)
                else:
                    rel_position, rel_direction, rel_dist = self.bev_sampled_rays
                    ray_xyz = (R @ rel_position.reshape((-1,3)).T + T).T

                    ray_xyz = torch.tensor(ray_xyz,dtype=torch.float32,device=self.device).view(self.args.localization_map_height*self.args.localization_map_width,self.args.bev_N_samples,3)

                with torch.no_grad():
                    occupancy_query = ray_xyz.view(-1,3)
                    searched_occupancy_dists, searched_occupancy_inds = occupancy_fcd_tree.query(occupancy_query, nr_nns_searches=1) #Note that the cupy_kdtree distances are squared
                    occupancy_masks = (searched_occupancy_dists < self.args.bev_feature_fields_search_radius).view(-1,)
                    occupancy_ray_xyz = ray_xyz.view(-1,3)[occupancy_masks]

                    occupancy_ray_k_neighbor_dists, occupancy_ray_k_neighbor_inds = fcd_tree.query(occupancy_ray_xyz, nr_nns_searches=self.args.bev_feature_fields_search_num)

                searched_ray_k_neighbor_dists = torch.full((ray_xyz.shape[0]*ray_xyz.shape[1],self.args.bev_feature_fields_search_num),self.args.bev_feature_fields_search_radius,dtype=torch.float32, device=self.device)
                searched_ray_k_neighbor_dists[occupancy_masks] = occupancy_ray_k_neighbor_dists

                searched_ray_k_neighbor_inds = torch.full((ray_xyz.shape[0]*ray_xyz.shape[1],self.args.bev_feature_fields_search_num),-1,dtype=torch.int64,device=self.device)
                searched_ray_k_neighbor_inds[occupancy_masks] = occupancy_ray_k_neighbor_inds

                searched_ray_k_neighbor_dists = torch.sqrt(searched_ray_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
                searched_ray_k_neighbor_inds[searched_ray_k_neighbor_dists >= self.args.bev_feature_fields_search_radius] = -1
                searched_ray_k_neighbor_dists[searched_ray_k_neighbor_dists >= self.args.bev_feature_fields_search_radius] = self.args.bev_feature_fields_search_radius


                searched_ray_k_neighbor_inds = searched_ray_k_neighbor_inds.view(ray_xyz.shape[0],ray_xyz.shape[1],self.args.bev_feature_fields_search_num)
                searched_ray_k_neighbor_dists = searched_ray_k_neighbor_dists.view(ray_xyz.shape[0],ray_xyz.shape[1],self.args.bev_feature_fields_search_num)

                tmp_distance = searched_ray_k_neighbor_dists.sum(-1)
                tmp_density = (1/tmp_distance)                
                topk_output = torch.topk(tmp_density, k=self.args.bev_N_importance, dim=-1, largest=True)[1]
                topk_inds = torch.sort(topk_output, dim=-1)[0] # Search important sampled points

                sample_ray_xyz = torch.gather(ray_xyz,1,topk_inds.unsqueeze(-1).repeat(1,1,3))
                sample_ray_direction = rel_direction[::,-1]

                if self.gt_pcd_tree != None and self.gt_pcd_tree[batch_id] != None:
                    gt_ray_xyz = torch.gather(ray_xyz,1,topk_output.unsqueeze(-1).repeat(1,1,3))[...,0,:]
                    with torch.no_grad():
                        gt_dists, gt_inds = self.gt_pcd_tree[batch_id].query(gt_ray_xyz,nr_nns_searches=1)
                    gt_dists = torch.sqrt(gt_dists) #Note that the cupy_kdtree distances are squared
                    gt_label = self.gt_pcd_label[batch_id][gt_inds]
                    gt_label[gt_dists >= self.args.bev_feature_fields_search_radius] = -100
                    gt_label[searched_ray_k_neighbor_inds.sum(-1).sum(-1) == -self.args.bev_N_samples*self.args.bev_feature_fields_search_num] = -100
                    batch_gt_label.append(gt_label)

                elif self.gt_pcd_tree != None and self.gt_pcd_tree[batch_id] == None:
                    batch_gt_label.append(None)

                if visualization: # Visualize the feature points of feature fields and sampled points of rays, set `True` for debug
                    rays_pcd=o3d.geometry.PointCloud()
                    rays_pcd.points = o3d.utility.Vector3dVector(sample_ray_xyz.view(-1,3).cpu().numpy()) # ray_xyz, sample_ray_xyz
                    feature_pcd=o3d.geometry.PointCloud()
                    feature_pcd.points = o3d.utility.Vector3dVector( np.concatenate([self.global_position_x[batch_id].reshape((-1,1)),self.global_position_y[batch_id].reshape((-1,1)),self.global_position_z[batch_id].reshape((-1,1))],axis=-1) )
                    #gt_pcd=o3d.geometry.PointCloud()
                    #gt_pcd.points = o3d.utility.Vector3dVector(self.gt_pcd_xyz[batch_id])
                    #feature_pcd += gt_pcd
                    feature_pcd += rays_pcd
                    o3d.visualization.draw_geometries([feature_pcd])

                with torch.no_grad():
                    sample_feature_k_neighbor_dists, sample_feature_k_neighbor_inds = fcd_tree.query(sample_ray_xyz.view(-1,3), nr_nns_searches=self.args.bev_feature_fields_search_num)

                sample_feature_k_neighbor_dists = torch.sqrt(sample_feature_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
                sample_feature_k_neighbor_inds[sample_feature_k_neighbor_dists >= self.args.bev_feature_fields_search_radius] = -1 # mask the unnecessary features
                sample_feature_k_neighbor_inds = sample_feature_k_neighbor_inds.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],self.args.bev_feature_fields_search_num)

                sample_ft_neighbor_xyzds = torch.zeros((sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],self.args.bev_feature_fields_search_num,6),dtype=torch.float32, device=self.device)
                filtered_points_mask = sample_feature_k_neighbor_inds.to(torch.float32).mean(-1) # filter out the unnecessary points, whose value is -1
                filtered_points_mask = filtered_points_mask!=-1
                selected_fts_idx = sample_feature_k_neighbor_inds 
                sample_ft_neighbor_xyzds[...,:3] = fcd_points[selected_fts_idx] - sample_ray_xyz.unsqueeze(-2)
                # Get the relative positions of features to the sampled point


                sample_ft_neighbor_x = sample_ft_neighbor_xyzds[...,0]
                sample_ft_neighbor_y = sample_ft_neighbor_xyzds[...,1]
                sample_ft_neighbor_xyzds[...,0] = sample_ft_neighbor_x * math.cos(-camera_direction) - sample_ft_neighbor_y * math.sin(-camera_direction)
                sample_ft_neighbor_xyzds[...,1] = sample_ft_neighbor_x * math.sin(-camera_direction) + sample_ft_neighbor_y * math.cos(-camera_direction)
            
                sample_ft_neighbor_xyzds[...,:3][selected_fts_idx==-1] = self.args.far

                sample_ray_direction = torch.tensor(sample_ray_direction,device=sample_ft_neighbor_xyzds.device)

                sample_ray_direction =  patch_directions[selected_fts_idx] - sample_ray_direction.unsqueeze(-1).unsqueeze(-1)
                sample_ray_direction = torch.cat((torch.sin(sample_ray_direction).unsqueeze(-1), torch.cos(sample_ray_direction).unsqueeze(-1)), dim=-1)

                sample_ft_neighbor_xyzds[...,3:5] = sample_ray_direction
                sample_ft_neighbor_xyzds[...,3:5][selected_fts_idx==-1] = 0
                sample_ft_neighbor_xyzds[...,5:] = patch_scales[selected_fts_idx]
                sample_ft_neighbor_xyzds[...,5:][selected_fts_idx==-1] = 0
                sample_ft_neighbor_embedding = scene_fts[selected_fts_idx.cpu().numpy()]
                sample_ft_neighbor_embedding = torch.tensor(sample_ft_neighbor_embedding,dtype=torch.float16,device=self.device)
                sample_ft_neighbor_embedding[selected_fts_idx==-1] = 0 # mask the unnecessary features
                sample_feature = torch.zeros((sample_ft_neighbor_embedding.shape[0],sample_ft_neighbor_embedding.shape[1], sample_ft_neighbor_embedding.shape[-1]),dtype=torch.float16, device=self.device)
                sample_density = torch.zeros((sample_ft_neighbor_embedding.shape[0],sample_ft_neighbor_embedding.shape[1]),dtype=torch.float16,device=self.device)
                sample_ft_neighbor_embedding = sample_ft_neighbor_embedding[filtered_points_mask] # filter out the unnecessary points, whose idx value is -1
                sample_ft_neighbor_xyzds = sample_ft_neighbor_xyzds[filtered_points_mask] # filter out the unnecessary points, whose idx value is -1

                if sample_ft_neighbor_embedding.shape[0] != 0 :
                    sample_feature[filtered_points_mask], sample_density[filtered_points_mask] = self.run_bev_region_encode(sample_ft_neighbor_embedding, sample_ft_neighbor_xyzds)
                else: # if sample_ft_neighbor_embedding.shape[0] == 0, input zero value to avoid DDP error, "Use of a module parameter outside the `forward` function"
                    sample_feature[0], _ = self.run_bev_region_encode(torch.zeros((1,self.args.mlp_net_width*self.args.feature_fields_search_num),dtype=torch.float16,device=self.device), torch.zeros((self.args.feature_fields_search_num,6),dtype=torch.float16,device=self.device))
            
                feature_map = self.bev_raw2feature(sample_feature, sample_density)
                batch_bev_feature_map.append(feature_map.unsqueeze(0))

        batch_bev_feature_map = torch.cat(batch_bev_feature_map,dim=0)
        batch_bev_fts = batch_bev_feature_map.view(self.batch_size,self.args.localization_map_height,self.args.localization_map_width,self.args.bev_mlp_net_width)
        batch_bev_fts = self.bev_enc_conv(batch_bev_fts.permute(0,3,1,2)).permute(0,2,3,1).view(self.batch_size,self.args.grid_map_height * self.args.grid_map_width,self.args.bev_mlp_net_width) # Conv to downsample the bev map
        batch_bev_fts = self.bev_encoder(batch_bev_fts + self.bev_positional_embedding)

        return batch_bev_fts, batch_bev_feature_map, batch_gt_label



    def get_kd_tree(self):

        occupancy_unit_size = self.args.occupancy_unit_size
        batch_fcd = []
        batch_fcd_tree = []
        batch_occupancy_fcd_tree = []
        for i in range(self.batch_size):

            fcd =  torch.tensor(np.concatenate((self.global_position_x[i].reshape((-1,1)),self.global_position_y[i].reshape((-1,1)),self.global_position_z[i].reshape((-1,1))),axis=-1),dtype=torch.float32, device=self.device)
            fcd_tree = build_kd_tree(fcd)
            batch_fcd.append(fcd)
            batch_fcd_tree.append(fcd_tree)

            # Feature fields occupancy
            occupancy_fcd = torch.div(fcd, occupancy_unit_size, rounding_mode='trunc')
            occupancy_fcd = torch.unique(occupancy_fcd, dim=0) * occupancy_unit_size
            occupancy_fcd_tree = build_kd_tree(occupancy_fcd)
            batch_occupancy_fcd_tree.append(occupancy_fcd_tree)
       
        return batch_fcd, batch_fcd_tree, batch_occupancy_fcd_tree


    def get_heading_angle(self, position):      
        dx = position[:,0]
        dy = position[:,1]
        dz = position[:,2]
        xy_dist = np.sqrt(np.square(dx) + np.square(dy))
        xy_dist[xy_dist < 1e-4] = 1e-4
        # the simulator's api is weired (x-y axis is transposed)
        heading_angle = - np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
        heading_angle[dy < 0] =  heading_angle[dy < 0] - np.pi
        return heading_angle


    def project_depth_to_3d_habitat(self, depth_map, heading_angle):
        W = self.args.input_width
        H = self.args.input_height
        half_W = W//2
        half_H = H//2
        depth_y = depth_map.astype(np.float32) # / 4000.

        tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi * self.args.input_hfov/360.)
        direction = - np.arctan(tan_xy)
        depth_x = depth_y * tan_xy
        depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi * self.args.input_vfov/360.))
        scale = depth_y * math.tan(math.pi * self.args.input_hfov/360.) * 2. / W

        direction = (direction+heading_angle) % (2*math.pi)
        rel_x = depth_x * math.cos(heading_angle) - depth_y * math.sin(heading_angle)
        rel_y = depth_x * math.sin(heading_angle) + depth_y * math.cos(heading_angle)
        rel_z = depth_z
        return rel_x, rel_y, rel_z, direction.reshape(-1), scale.reshape(-1)


    def update_feature_fields_habitat(self, batch_position, batch_heading, batch_depth, batch_grid_ft, num_of_views=12):

        for i in range(self.batch_size):
            position = batch_position[i].copy()
            position[0], position[1], position[2] = batch_position[i][0], - batch_position[i][2], batch_position[i][1] # Note to swap y,z axis, - y
            heading = batch_heading[i]
            depth = batch_depth[i]
            grid_ft = batch_grid_ft[i]
            viewpoint_x_list = []
            viewpoint_y_list = []
            viewpoint_z_list = []
            viewpoint_scale_list = []
            viewpoint_direction_list = []

            depth = depth.reshape((-1,self.args.input_height*self.args.input_width))
       
            for ix in range(num_of_views): # 12 views, rotation for panorama
                rel_x, rel_y, rel_z, direction, scale = self.project_depth_to_3d_habitat(depth[ix:ix+1],ix*(-math.pi/6)+heading)  
                global_x = rel_x + position[0]
                global_y = rel_y + position[1]
                global_z = rel_z + position[2]

                viewpoint_x_list.append(global_x)
                viewpoint_y_list.append(global_y)
                viewpoint_z_list.append(global_z)
                viewpoint_scale_list.append(scale)
                viewpoint_direction_list.append(direction)

            width = self.args.mlp_net_width
            if self.global_fts[i] == []:
                self.global_fts[i] = grid_ft.reshape((-1,width))          
            else:
                self.global_fts[i] = np.concatenate((self.global_fts[i],grid_ft.reshape((-1,width))),axis=0)

            position_x = np.concatenate(viewpoint_x_list,0)
            position_y = np.concatenate(viewpoint_y_list,0)
            position_z = np.concatenate(viewpoint_z_list,0)
            patch_scales = np.concatenate(viewpoint_scale_list,0)
            patch_directions = np.concatenate(viewpoint_direction_list,0)

            if self.global_position_x[i] == []:
                self.global_position_x[i] = position_x
                self.global_position_y[i] = position_y
                self.global_position_z[i] = position_z
                self.global_patch_scales[i] = patch_scales
                self.global_patch_directions[i] = patch_directions
            else:
                self.global_position_x[i] = np.concatenate([self.global_position_x[i],position_x],0)
                self.global_position_y[i] = np.concatenate([self.global_position_y[i],position_y],0)
                self.global_position_z[i] = np.concatenate([self.global_position_z[i],position_z],0)
                self.global_patch_scales[i] = np.concatenate([self.global_patch_scales[i],patch_scales],0)
                self.global_patch_directions[i] = np.concatenate([self.global_patch_directions[i],patch_directions],0)

        self.fcd, self.fcd_tree, self.occupancy_fcd_tree = self.get_kd_tree()


    def update_feature_fields(self, batch_depth, batch_grid_ft, batch_camera_intrinsic, batch_rot, batch_trans, depth_scale=1000.0, depth_trunc=1000.0):

        for i in range(self.batch_size):
            thread_output = self.thread_pool([ [project_depth_to_3d, [ batch_depth[i][job_id], batch_camera_intrinsic[i][job_id], batch_rot[i][job_id], batch_trans[i][job_id],depth_scale,depth_trunc,self.args.input_height,self.args.input_width ], {} ] for job_id in range(len(batch_depth[i])) ]) # Parallel computing with multiple CPUs

            for j in range(batch_depth[i].shape[0]):
                points, points_mask = thread_output[j]
                points = points[points_mask] # filter out the noise
                rel_position, rel_direction, rel_dist = self.sampled_rays
                patch_scales = points[:,-1] * math.fabs(math.tan(rel_direction[0][-1])) * 2. / self.args.view_width
                R = batch_rot[i][j]
                T = batch_trans[i][j]
                points = (R @ points.T + T).T  

                patch_directions = self.get_heading_angle(points)


                width = self.args.mlp_net_width
                grid_ft = batch_grid_ft[i][j]
                grid_ft = grid_ft.reshape((-1,width))[points_mask] # filter out the noise
                if self.global_fts[i] == []:
                    self.global_fts[i] = grid_ft       
                else:
                    self.global_fts[i] = np.concatenate((self.global_fts[i],grid_ft),axis=0)

                if self.global_position_x[i] == []:
                    self.global_position_x[i] = points[:,0]
                    self.global_position_y[i] = points[:,1]
                    self.global_position_z[i] = points[:,2]
                    self.global_patch_scales[i] = patch_scales
                    self.global_patch_directions[i] = patch_directions
                else:
                    self.global_position_x[i] = np.concatenate([self.global_position_x[i],points[:,0]],0)
                    self.global_position_y[i] = np.concatenate([self.global_position_y[i],points[:,1]],0)
                    self.global_position_z[i] = np.concatenate([self.global_position_z[i],points[:,2]],0)
                    self.global_patch_scales[i] = np.concatenate([self.global_patch_scales[i],patch_scales],0)
                    self.global_patch_directions[i] = np.concatenate([self.global_patch_directions[i],patch_directions],0)

        self.fcd, self.fcd_tree, self.occupancy_fcd_tree = self.get_kd_tree()
            