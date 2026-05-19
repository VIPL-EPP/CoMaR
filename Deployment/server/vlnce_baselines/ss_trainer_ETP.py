import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
import tqdm
from gym import Space
from habitat import Config, logger

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST, calculate_vp_rel_pos_fts
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from fastdtw import fastdtw

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
import cv2
from PIL import Image
import vlnce_baselines.waypoint_networks.utils as utils
from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele )
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from vlnce_baselines.waypoint_networks.semantic_grid import SemanticGrid
from vlnce_baselines.waypoint_networks import get_img_segmentor_from_options
from vlnce_baselines.waypoint_networks.resnetUnet import ResNetUNet
import vlnce_baselines.waypoint_networks.viz_utils as viz_utils
import matplotlib.pyplot as plt
from datetime import timedelta

from vlnce_baselines.models.Policy_ViewSelection_ETP import Net
from transformers import AutoTokenizer

import cv2
from scipy.spatial.transform import Rotation as R

from fastapi import FastAPI
from pydantic import BaseModel, Field
import threading
import uvicorn
import base64
from typing import Optional, Any

from vlnce_baselines.pret_utils.mask import length_to_mask

class RLTrainer():
    def __init__(self, config=None):
        self.device = 'cuda'
        self.max_len = int(config['IL']['max_traj_len']) #  * 0.97 transfered gt path got 0.96 spl
        self.config = config
        self.batch_size = 1

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

        
    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
    ):
    
        self.policy = Net(config['MODEL'])
        
        self.policy.net.vln_bert.feature_fields.load_state_dict(torch.load("/data/3dg-LF/3dff.pth"),strict=True)

        self.tokenizer = AutoTokenizer.from_pretrained('bert_config/bert-base-uncased')

        n_object_classes = 27

        ## Load the pre-trained img segmentation model
        self.img_segmentor = get_img_segmentor_from_options(n_object_classes,1.0)
        self.img_segmentor = self.img_segmentor.to(self.device)

        self.img_segmentor = torch.nn.DataParallel(self.img_segmentor)

        checkpoint = torch.load("pretrained/segm.pt")
        self.img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'])         
        self.img_segmentor.eval()

        self.policy.net.occupancy_map_predictor = ResNetUNet(3,3,True)
        self.policy.net.semantic_map_predictor = ResNetUNet(n_object_classes+3,n_object_classes,True)
        self.policy.net.waypoint_predictor = ResNetUNet(n_object_classes+3,1,True)
        
        self.policy.net.rgb_encoder.eval()
        self.policy.net.occupancy_map_predictor.eval()
        self.policy.net.semantic_map_predictor.eval()
        self.policy.net.waypoint_predictor.eval()
        
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        # self.noise_filter = torch.nn.Conv2d(1, 1, (7, 7), padding=(3,3)).to(self.device)
        # noise_filter_weight = torch.ones(1,1,7,7).to(self.device) #/ (7.*7.)
        # self.noise_filter.weight = torch.nn.Parameter(noise_filter_weight)
        # self.noise_filter.eval()

        self.img_segm_size = (128,128)
        ## Build necessary info for ground-projecting the semantic segmentation
        self._xs, self._ys = torch.tensor(np.array(np.meshgrid(np.linspace(-1,1,self.img_segm_size[0]), np.linspace(1,-1,self.img_segm_size[1]))), device=self.device)
        self._xs = self._xs.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        self._ys = self._ys.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        _x, _y = torch.tensor(np.array(np.meshgrid(np.linspace(0, self.img_segm_size[0]-1, self.img_segm_size[0]), 
                                                    np.linspace(0, self.img_segm_size[1]-1, self.img_segm_size[1]))), device=self.device)
        _xy_img = torch.cat((_x.reshape(1,self.img_segm_size[0],self.img_segm_size[1]), _y.reshape(1,self.img_segm_size[0],self.img_segm_size[1])), dim=0)
        _points2D_step = _xy_img.reshape(2, -1)
        self._points2D_step = torch.transpose(_points2D_step, 0, 1) # Npoints x 2  

        self.policy.to(self.device)
        self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
            device_ids=[self.device], output_device=self.device)
            
        ckpt_dict = self.load_checkpoint('pretrained/cwp_predictor.pth', map_location="cpu")           
        b = [key for key in ckpt_dict["state_dict"].keys()]
        for key in b:
            if 'rgb_encoder' in key:
                ckpt_dict['state_dict'].pop(key) 
        self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
            
        if load_from_ckpt:
            ckpt_dict = self.load_checkpoint(config['IL']['ckpt_to_load'], map_location="cpu")           
            self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
            start_iter = ckpt_dict["iteration"]

        # ===== FastAPI init =====
        self._init_fastapi()

        return start_iter
    
    class ProcessRequest(BaseModel):
        timestamp: str = Field(..., description="sec.nanosec string")
        rgb_image: str = Field(..., description="jpg base64")
        depth_image: str = Field(..., description="png base64 (likely uint16)")
        odometry: dict = Field(..., description="odom dict")

    def _init_fastapi(self):
        """
        适配发送端:
        - POST /process
            输入: {timestamp, rgb_image(jpg b64), depth_image(png b64), odometry}
            输出: subgoal/stop 等 dict
        """
        self.app = FastAPI(title="3DFF-RLTrainer-Server")

        # ---- observation slot ----
        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()
        self._latest_obs: Optional[dict] = None

        # ---- response slot (per request) ----
        self._resp_lock = threading.Lock()
        self._pending_resp_event: Optional[threading.Event] = None
        self._pending_resp_payload: Optional[dict] = None
        self._pending_timestamp: Optional[str] = None

        # ---- latest subgoal cache (optional) ----
        self._subgoal_lock = threading.Lock()
        self._latest_subgoal: Optional[dict] = None

        @self.app.post("/process")
        def process(req: RLTrainer.ProcessRequest):
            # 1) decode images + odom -> location
            try:
                rgb_np = self._decode_rgb_b64(req.rgb_image)
                depth_np = self._decode_depth_b64(req.depth_image)
                location = self._odom_to_location(req.odometry)
            except Exception as e:
                return {"timestamp": req.timestamp, "status": "error", "error": str(e)}

            # 2) register this request's response waiter
            resp_event = threading.Event()
            with self._resp_lock:
                self._pending_resp_event = resp_event
                self._pending_resp_payload = None
                self._pending_timestamp = req.timestamp

            # 3) store obs and wake rollout
            with self._obs_lock:
                self._latest_obs = {
                    "timestamp": req.timestamp,
                    "rgb_np": rgb_np,
                    "depth_np": depth_np,
                    "location": location,
                }
                self._obs_event.set()

            # 4) block until rollout publishes a response
            resp_event.wait()  # 发送端 timeout=None，这里也不主动超时

            with self._resp_lock:
                payload = self._pending_resp_payload
                # 清理当前 pending，避免下次误用
                self._pending_resp_event = None
                self._pending_resp_payload = None
                self._pending_timestamp = None

            # 防御式
            if payload is None:
                payload = {"timestamp": req.timestamp, "status": "error", "error": "empty response"}

            return payload

        # 端口默认对齐你的发送端
        host = "0.0.0.0"
        port = 5000
        if isinstance(self.config, dict):
            port = int(self.config.get("FASTAPI_PORT", 5000))

        self._fastapi_thread = threading.Thread(
            target=self._run_fastapi,
            args=(host, port),
            daemon=True
        )
        self._fastapi_thread.start()

    def _run_fastapi(self, host: str, port: int):
        # 单 worker 避免多进程状态不同步
        uvicorn.run(self.app, host=host, port=port, log_level="warning", workers=1)

    # ---------- decode helpers ----------

    def _decode_rgb_b64(self, b64_str: str) -> np.ndarray:
        data = base64.b64decode(b64_str)
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode rgb jpg")
        # 发送端是 bgr8 -> jpg，所以这里转成 RGB 更安全
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _decode_depth_b64(self, b64_str: str) -> np.ndarray:
        data = base64.b64decode(b64_str)
        arr = np.frombuffer(data, dtype=np.uint8)
        depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError("Failed to decode depth png")
        return depth.astype(np.float32)

    def _odom_to_location(self, odom_dict: dict) -> dict:
        # 只取你 transform_pose 需要的字段
        pos = odom_dict.get("position", {})
        ori = odom_dict.get("orientation", {})
        return {
            "position": {
                "x": float(pos.get("x", 0.0)),
                "y": float(pos.get("y", 0.0)),
                "z": float(pos.get("z", 0.0)),
            },
            "orientation": {
                "x": float(ori.get("x", 0.0)),
                "y": float(ori.get("y", 0.0)),
                "z": float(ori.get("z", 0.0)),
                "w": float(ori.get("w", 1.0)),
            }
        }

    # ---------- rollout side wait/response ----------

    def _wait_for_observation_from_api(self):
        """
        rollout 内部调用:
        等待 /process 写入最新观测
        """
        self._obs_event.wait()
        self._obs_event.clear()

        with self._obs_lock:
            obs = self._latest_obs

        if obs is None:
            raise RuntimeError("Observation event set but no obs payload found.")

        # 转 torch / numpy
        img = torch.as_tensor(obs["rgb_np"])  # H W 3, uint8
        depth = obs["depth_np"]               # H W, float32
        location = obs["location"]
        timestamp = obs["timestamp"]

        return img, depth, location, timestamp

    def _publish_subgoal_response(self, subgoal: Any, status: str, timestamp: Optional[str] = None):
        """
        在 rollout 的注释位置调用:
        - 更新缓存
        - 若 /process 正在等待，则立刻唤醒并返回
        """
        payload = {
            "timestamp": timestamp,
            "status": status,
            "subgoal": subgoal
        }

        # cache
        with self._subgoal_lock:
            self._latest_subgoal = deepcopy(payload)

        # reply to pending /process
        with self._resp_lock:
            if self._pending_resp_event is not None:
                # 如果没显式传 timestamp，就用当前 pending 的
                if payload["timestamp"] is None:
                    payload["timestamp"] = self._pending_timestamp
                self._pending_resp_payload = deepcopy(payload)
                self._pending_resp_event.set()

    # ===================== FastAPI BRIDGE END =====================


    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_loc_fts = [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.batch_size):
            rgb_fts, loc_fts , nav_types = [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }


    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = len(cur_vp)
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }
    

    def get_MAM_variable(self, cur_vp, cur_ghost, txt_embeds, txt_masks):
        path_features = []
        local_features = []

        for i, graph in enumerate(self.gmaps):
            current_viewpoint_id = cur_vp[i]
            trajectory = graph.get_shortest_path(current_viewpoint_id)
            path_feature = [graph.Path_graph.edges[v, trajectory[i+1]]['feature'] for i, v in enumerate(trajectory[:-1])]
            path_feature = torch.stack(path_feature) \
                                if len(path_feature) > 0 \
                                else torch.zeros(0, 768, device=self.device)
            path_features.append(path_feature)

            local_feature = [graph.get_ghost_front_edges(v, current_viewpoint_id) for v in cur_ghost[i]]
            local_feature = torch.stack(local_feature) \
                                if len(local_feature) > 0 \
                                else torch.zeros(0, 768, device=self.device)
            local_features.append(local_feature)
        
        path_lengths = [f.shape[0] for f in path_features]
        path_padding_mask = length_to_mask(path_lengths, device=self.device)
        path_features = pad_sequence(path_features, batch_first=True)

        local_lengths = [f.shape[0] for f in local_features]
        local_padding_mask = length_to_mask(local_lengths, device=self.device)
        local_features = pad_sequence(local_features, batch_first=True)

        return {
            'path_features': path_features, 'path_padding_mask':path_padding_mask, 
            'local_features': local_features, 'local_padding_mask': local_padding_mask,
            'txt_embeds': txt_embeds, 'txt_masks': ~txt_masks
        }
    
    def get_CCM_variable(self, cur_path_vp):
        score_batch = []
        unvisited_batch = []
        batch_gmap_visited_masks = []
        for i, graph in enumerate(self.gmaps):
            node_vp_ids = list(graph.node_pos.keys())
            ghost_vp_ids = list(graph.ghost_pos.keys())
            current_viewpoint_id = cur_path_vp[i]
            unvisited_or_current = [current_viewpoint_id] + node_vp_ids + ghost_vp_ids
            unvisited_batch.append(unvisited_or_current)

            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            # score = [graph.get_path_embeds(v) for v in unvisited_or_current]
            score = []
            for v in unvisited_or_current:
                score.append(graph.get_path_embeds(v))
            score = torch.stack(score)  # (N,)
            score_batch.append(score)

            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
    
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()
        lengths = [s.shape[0] for s in score_batch]
        padding_mask = length_to_mask(lengths, device=self.device)
        visited_mask = batch_gmap_visited_masks
        score_batch = pad_sequence(score_batch, batch_first=True)  # (B, N)
        return {'unvisited_or_current': unvisited_batch, 
                'path_feature': score_batch, 
                'padding_mask': padding_mask, 
                "visited_mask": visited_mask}


    
    def first_frame_coordinate(self, pose_data):
        import quaternion
        q = np.quaternion(pose_data['orientation']['w'], pose_data['orientation']['x'], 
                          pose_data['orientation']['y'], pose_data['orientation']['z'])
        rotation_matrix = quaternion.as_rotation_matrix(q)
        T = np.eye(4)
        T[0:3, 0:3] = rotation_matrix
        T[0:3, 3] = [pose_data['position']['x'], pose_data['position']['y'], pose_data['position']['z']]

        T_rel = None

        if self.T_first is None:
            self.T_first = T
            T_rel = np.eye(4)
            self.position_first = np.array([pose_data['position']['x'], pose_data['position']['y'], pose_data['position']['z']])
            self.orientation_first = [pose_data['orientation']['x'],pose_data['orientation']['y'], pose_data['orientation']['z'],pose_data['orientation']['w'],]
        else:
            T_rel = np.linalg.inv(self.T_first) @ T
        
        q = quaternion.from_rotation_matrix(T_rel[:3, :3])
        transformed_data = {
            'position': {
                'x': T_rel[0, 3],
                'y': T_rel[1, 3],
                'z': T_rel[2, 3]
            },
            'orientation': {
                'x': q.x,
                'y': q.y,
                'z': q.z,
                'w': q.w
            }
        }
        return transformed_data
    

    def cood_align(self, pose_data, T=np.array([[0, -1, 0],[0, 0, 1],[-1, 0, 0]])):
        # --- 处理位置 (Position) ---
        original_pos = np.array([
            pose_data['position']['x'],
            pose_data['position']['y'],
            pose_data['position']['z']
        ])
        # 矩阵乘法: P_new = T * P_old
        new_pos = T @ original_pos
        # --- 处理姿态 (Orientation) ---
        original_quat = [
            pose_data['orientation']['x'],
            pose_data['orientation']['y'],
            pose_data['orientation']['z'],
            pose_data['orientation']['w']
        ]

        # 欧拉角
        r_old = R.from_quat(original_quat)
        euler_old = r_old.as_euler('xyz', degrees=False)
        # 转轴位置变换即可 按照T矩阵变换
        euler_new = np.zeros(3)
        euler_new[0] = euler_old[1]
        euler_new[1] = euler_old[2]
        euler_new[2] = euler_old[0]

        r_new = R.from_euler('xyz', euler_new, degrees=False)
        new_quat = r_new.as_quat() # 返回格式为 [x, y, z, w]


        # # 将四元数转换为旋转矩阵 R_old
        # r_old = R.from_quat(original_quat)
        # matrix_old = r_old.as_matrix()
        # # 应用变换: R_new = T * R_old
        # # 物理含义: 先把物体转成原来的姿态，然后再把整个坐标系按 T 变换
        # matrix_new = T @ matrix_old
        # # 将新矩阵转换回四元数
        # r_new = R.from_matrix(matrix_new)
        # new_quat = r_new.as_quat() # 返回格式为 [x, y, z, w]

        # --- 组装结果 ---
        transformed_data = {
            'position': {
                'x': new_pos[0],
                'y': new_pos[1],
                'z': new_pos[2]
            },
            'orientation': {
                'x': new_quat[0],
                'y': new_quat[1],
                'z': new_quat[2],
                'w': new_quat[3]
            }
        }
        
        return transformed_data
    
    def transform_pose(self, pose_data):
        transformed_data = self.first_frame_coordinate(pose_data)
        transformed_data = self.cood_align(transformed_data)
        head_vector = self.calculate_head_vec(transformed_data['orientation'])
        # print(head_vector)
        return transformed_data, head_vector
    
    def world_coordinate(self, position):
        import quaternion
        
        r_wl = R.from_quat(self.orientation_first)
        
        t_wl = self.position_first

        # p_B = R_BA * p_A + t_BA
        p_B = r_wl.apply(position) + t_wl

        transformed_data = {
            'position': p_B.tolist()
        }
        return transformed_data
    
    def cood_align_re(self, original_pos, T=np.array([[ 0,  0, -1],[-1,  0,  0],[ 0,  1,  0]])):
        # 矩阵乘法: P_new = T * P_old
        new_pos = T @ original_pos.transpose()

        return new_pos.transpose()
    
    
    
    def pack_action(self, cpos, gpos):
        pose_data = np.array([[cpos[0], cpos[1], cpos[2]], [gpos[0], gpos[1], gpos[2]]])
        transformed_data = self.cood_align_re(pose_data, T=np.array([[ 0,  0, -1],[-1,  0,  0],[ 0,  1,  0]]))
        print(f"first frame path coordinate: {transformed_data}")
        transformed_data = self.world_coordinate(transformed_data)
        print(f"world path coordinate: {transformed_data} \n")
        return transformed_data


    def quaternion_rotate_vector(self, quat: np.quaternion, v: np.array) -> np.array:
        r"""Rotates a vector by a quaternion
        Args:
            quaternion: The quaternion to rotate by
            v: The vector to rotate
        Returns:
            np.array: The rotated vector
        """
        vq = np.quaternion(0, 0, 0, 0)
        vq.imag = v
        return (quat * vq * quat.inverse()).imag
    
    def cartesian_to_polar(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    def calculate_head_vec(self, orientation):
        # 四元数到旋转矩阵
        quat = np.quaternion(orientation['w'], orientation['x'], orientation['y'], orientation['z'])
        heading_vector = self.quaternion_rotate_vector(
            quat.inverse(), np.array([0, 0, -1])
        )
        heading = self.cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return heading  # 单位化
    

    def calculate_distance_xy(self, pos1, pos2):
        # pos1 and pos2 are np.array with shape (3,)
        xy_cur = np.array([pos1[0], pos1[1]])
        xy_tar = np.array([pos2[0], pos2[1]])
        return np.linalg.norm(xy_cur - xy_tar)
    

    def _send_rotate_observation_subgoal(self, cur_pos, observe_environment, timestamp):
        """
        observation_env < 9 时：
        每步向左旋转 45°，仅发送坐标（不依赖真实位移）
        """

        # 初始参数
        init_y = 0.33
        radius = math.sqrt(2.0)

        # env=1 -> 45°
        angle_deg = observe_environment * 45.0
        angle_rad = math.radians(angle_deg)

        # 左旋（逆时针），z 向后为负
        x = -math.sin(angle_rad) * radius
        z = -math.cos(angle_rad) * radius
        y = init_y

        target_pos = np.array([x, y, z])

        print(
            f"[Observation Rotate] env={observe_environment}, "
            f"angle={angle_deg:.1f}°, send pos={target_pos.tolist()}"
        )

        # 构造 subgoal（起点终点一样即可，前端不考虑位移）
        subgoal = self.pack_action(
            cpos=cur_pos,
            gpos=target_pos
        )

        # 发送给前端
        self._publish_subgoal_response(
            subgoal=subgoal,
            status="observe_rotate",
            timestamp=timestamp
        )

    


    def rollout(self, mode, instruction):
        # encode instructions
        all_txt_ids = self.tokenizer(instruction)['input_ids']
        all_txt_ids = torch.tensor([all_txt_ids])
        instr_pad_id = 1 if self.config['MODEL']['task_type'] == 'rxr' else 0
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        total_actions = 0.
        not_done_index = list(range(self.batch_size))
        have_real_pos = False
        ghost_aug = 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config['IL']['loc_noise'], 
                               self.config['MODEL']['merge_ghost'],
                               ghost_aug) for _ in range(self.batch_size)]
        prev_vp = [None] * self.batch_size
        prev_ghost = [None] * self.batch_size
        prev_ghost_emb = [None] * self.batch_size


        ##############
        loss = 0.
        total_actions = 0.

        # 读取出来的参数
        fx = 606.53
        fy = 321.41
        W = 640 # 假设是 raw image width
        H = 480

        # # 读取出来的参数
        # fx = 909.80
        # fy = 642.11
        # W = 1280 # 假设是 raw image width
        # H = 720

        # 动态计算真实 FOV
        hfov = 2 * np.arctan(W / (2 * fx))
        vfov = 2 * np.arctan(H / (2 * fy))

        map_config={'hfov':hfov,'vfov':vfov,'global_dim':(512,512),'grid_dim':(192,192),'heatmap_size':192,'cell_size':0.05,'img_segm_size':(128,128),'spatial_labels':3,'object_labels':27,'img_size':[256,256],'occupancy_height_thresh':-1.0,'norm_depth':True}
        # 3d info
        xs, ys = torch.tensor(np.array(np.meshgrid(np.linspace(-1,1,map_config['img_size'][0]), np.linspace(1,-1,map_config['img_size'][1]))), device=self.device)

        xs = xs.reshape(1,map_config['img_size'][0],map_config['img_size'][1])
        ys = ys.reshape(1,map_config['img_size'][0],map_config['img_size'][1])
        K = np.array([
            [1 / np.tan(map_config['hfov'] / 2.), 0., 0., 0.],
            [0., 1 / np.tan(map_config['vfov'] / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        inv_K = torch.tensor(np.linalg.inv(K), device=self.device)


        # For each episode we need a new instance of a fresh global grid
        sg_map_global = SemanticGrid(self.batch_size, map_config['global_dim'], map_config['heatmap_size'], map_config['cell_size'],
                            spatial_labels=map_config['spatial_labels'], object_labels=map_config['object_labels'])

        abs_poses = [[] for b in range(self.batch_size)]
        positions = [None for b in range(self.batch_size)]
        headings = [None for b in range(self.batch_size)]
        observations = [{} for b in range(self.batch_size)]
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module

        batch_size = self.batch_size
        wait_for_move = None
        observe_environment = 0
        collision_distance = 5.

        self.T_first = None
        self.subgoal = None
        self.world_subgoal = None

        for stepk in range(self.max_len):
            # agent's current position and heading
            observe_environment += 1
            cur_location_world = None
            for ob_i in range(batch_size):
                
                # FastAPI for receiving rgb, depth, location
                img, depth, location, ts = self._wait_for_observation_from_api()

                print(f"Current world coordinate: {location['position']}")
                cur_location_world=np.array([location['position']['x'], location['position']['y'], location['position']['z']])

                depth = np.array(depth, dtype=np.float32) * 0.001 # raw depth image is in mm
                depth[~np.isfinite(depth)] = 0
                depth[depth > 10] = 0.0
                depth = torch.tensor(depth)

               
                pos, head_vec = self.transform_pose(location)
                positions[ob_i] = np.array([pos['position']['x'], 0.33 ,pos['position']['z']]) # Robot height 0.88m
                headings[ob_i] = (2*math.pi+head_vec)%(2*math.pi)

                observations[ob_i]['rgb'] = F.interpolate(img.unsqueeze(0).permute(0,3,1,2), size=map_config['img_size'], mode='nearest').permute(0,2,3,1).squeeze(0)
                observations[ob_i]['depth'] =  F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=map_config['img_size'], mode='nearest').squeeze(0).squeeze(0)
                collision_distance = depth[depth>0.01].min().item()

                if stepk == 0:
                    policy_net.vln_bert.feature_fields.initialize_camera_setting(hfov=hfov, vfov=vfov)
                    policy_net.vln_bert.feature_fields.initialize_novel_view_setting(hfov=hfov, vfov=vfov)
                    policy_net.vln_bert.feature_fields.reset(batch_size)

            policy_net.action_step = stepk + 1
            policy_net.positions = positions
            policy_net.headings = [(heading+2*math.pi)%(2*math.pi) for heading in headings]

            with torch.no_grad():

                batch_img = []
                batch_depth = []
                batch_local3D_step = []
                batch_rel_abs_pose = []

                for b in range(batch_size):
                
                        ##################################
                    img = observations[b]['rgb'].to(self.device)
                    depth = observations[b]['depth'].to(self.device).to(torch.float32)
                    q1 = math.cos(headings[b]/2)
                    q2 = math.sin(headings[b]/2)
                    rotation = np.quaternion(q1,0,q2,0)
                    agent_state = {
                        'position': positions[b],
                        'rotation': rotation
                        }


                    ################
                    policy_net.positions[b] = positions[b] #!!!!!!!!!!!!!!!!!!!!
                    policy_net.headings[b] = headings[b]   #!!!!!!!!!!!!!!!!!!!!
                    ################

                    viz_img = img.cpu().numpy()
                    viz_depth = depth.cpu().numpy()
                    depth_abs = depth.unsqueeze(-1)

                    batch_img.append(img.unsqueeze(0))
                    batch_depth.append(depth_abs.unsqueeze(0))

                    local3D_step = utils.depth_to_3D(depth_abs, map_config['img_size'], xs, ys, inv_K)
                    batch_local3D_step.append(local3D_step)

                    agent_pose, y_height = utils.get_sim_location(agent_state=agent_state)

                    if len(abs_poses[b]) < stepk+1:
                        abs_poses[b].append(agent_pose)
                    else:
                        abs_poses[b][stepk] = agent_pose

                    # Keep track of the agent's relative pose from the initial position
                    rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[b][stepk], pos1=abs_poses[b][0])
                    _rel_abs_pose = torch.Tensor(rel_abs_pose).unsqueeze(0).to(torch.float16)
                    _rel_abs_pose = _rel_abs_pose.to(self.device)
                    batch_rel_abs_pose.append(_rel_abs_pose)

                if batch_rel_abs_pose != []:
                    ### Run the img segmentation model to get the ground-projected semantic segmentation
                    batch_abs_poses = torch.tensor(abs_poses).to(self.device)
                    batch_rel_abs_pose = torch.cat(batch_rel_abs_pose,dim=0)

                    batch_img = torch.cat(batch_img,dim=0)
                    
                    batch_depth = torch.cat(batch_depth,dim=0)
                    depth_img = batch_depth.clone().permute(0,3,1,2)

                    depth_img = F.interpolate(depth_img, size=map_config['img_segm_size'], mode='nearest')
                    imgData = utils.preprocess_img(batch_img, cropSize=map_config['img_segm_size'], pixFormat='NCHW', normalize=True)

                    segm_batch = {'images':imgData.to(self.device).unsqueeze(1),
                                'depth_imgs':depth_img.to(self.device).unsqueeze(1)}
                    
                    pred_ego_sseg, img_segm = utils.run_img_segm(model=self.img_segmentor, 
                                                            input_batch=segm_batch, 
                                                            object_labels=map_config['object_labels'], 
                                                            crop_size=map_config['global_dim'], 
                                                            cell_size=map_config['cell_size'],
                                                            xs=self._xs,
                                                            ys=self._ys,
                                                            inv_K=inv_K,
                                                            points2D_step=self._points2D_step)   

                    
                    # do ground-projection, update the projected map
                    ego_grid_sseg_3 = utils.est_occ_from_depth(batch_local3D_step, grid_dim=map_config['global_dim'], cell_size=map_config['cell_size'], 
                                                                                    device=self.device, occupancy_height_thresh=map_config['occupancy_height_thresh'])

                    # Transform the ground projected egocentric grids to geocentric using relative pose
                    occup_grid_sseg = sg_map_global.spatialTransformer(grid=ego_grid_sseg_3, pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                    semantic_grid_sseg = sg_map_global.spatialTransformer(grid=pred_ego_sseg[:,0], pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                    # step_geo_grid contains the map snapshot every time a new observation is added
                    global_step_occup_grid_sseg, global_step_segm_grid_sseg = sg_map_global.update_proj_grid_bayes(occup_grid_sseg.unsqueeze(1),semantic_grid_sseg.unsqueeze(1))

                self.policy.net(
                    mode = "feature_field",
                    observations = observations,
                    in_train = False,
                )

        

                # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
                step_occup_grid_sseg = sg_map_global.rotate_map(grid=global_step_occup_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                step_segm_grid_sseg = sg_map_global.rotate_map(grid=global_step_segm_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                # Crop the grid around the agent at each timestep
                step_occup_grid_maps = utils.crop_grid(grid=step_occup_grid_sseg, crop_size=map_config['grid_dim'])
                step_segm_grid_maps = utils.crop_grid(grid=step_segm_grid_sseg, crop_size=map_config['grid_dim'])               

                predicted_occup_grid_maps =  self.policy.net.module.occupancy_map_predictor(step_occup_grid_maps.unsqueeze(1))
                step_segm_occup_grid_maps = torch.cat((step_segm_grid_maps,predicted_occup_grid_maps),dim=-3)
                predicted_segm_grid_maps = self.policy.net.module.semantic_map_predictor(step_segm_occup_grid_maps.unsqueeze(1))
                step_segm_occup_grid_maps = torch.cat((predicted_segm_grid_maps.unsqueeze(1),predicted_occup_grid_maps.unsqueeze(1)),dim=-3)
                waypoint_grid_maps = self.policy.net.module.waypoint_predictor(step_segm_occup_grid_maps).view(batch_size,1,map_config['grid_dim'][0],map_config['grid_dim'][1]).squeeze(1)

                # 1. 定义保存路径 (建议放在循环外部，或者作为类的属性)
                save_dir = "/home/houjiewen/g3d/Server_g3d/output/" # 你可以修改为你想要的路径
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for b in range(self.batch_size):     
                    # 如果不需要实时弹窗显示，建议注释掉 plt.ion() 和 plt.pause()，并使用 plt.ioff()
                    plt.ion() 
                    plt.clf()
                    
                    # 设置画布大小，防止保存时图片太小挤在一起
                    fig = plt.gcf()
                    fig.set_size_inches(12, 10) # 设置宽12英寸，高10英寸

                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
                    
                    # --- 1. RGB ---
                    plt.subplot(2, 2, 1)
                    plt.imshow(viz_img)
                    plt.title('RGB Observation')
                    plt.axis('off') # 可选：关闭坐标轴刻度，图片更干净

                    # --- 2. Depth ---
                    plt.subplot(2, 2, 2)
                    # 确保 viz_depth 数据范围正确
                    if viz_depth.max() <= 1.0: 
                         viz_depth_vis = (viz_depth * 255).astype(np.uint8)
                    else:
                         viz_depth_vis = viz_depth.astype(np.uint8)
                    viz_depth_color = cv2.applyColorMap(viz_depth_vis, cv2.COLORMAP_JET)
                    # OpenCV是BGR，Matplotlib是RGB，需要转换颜色通道，否则颜色会怪怪的
                    viz_depth_color = cv2.cvtColor(viz_depth_color, cv2.COLOR_BGR2RGB)
                    plt.imshow(viz_depth_color)
                    plt.title('Depth Observation')
                    plt.axis('off')

                    # --- 3. Occupancy ---
                    occup_image = viz_utils.get_tensor_imgSegm(predicted_occup_grid_maps, labels=3)
                    plt.subplot(2, 2, 3)
                    plt.imshow(occup_image)
                    plt.title('Occupancy map')
                    plt.axis('off')

                    # --- 4. Semantic ---
                    semantic_image = viz_utils.get_tensor_imgSegm(predicted_segm_grid_maps, labels=27)
                    plt.subplot(2, 2, 4)
                    plt.imshow(semantic_image)
                    plt.title('Semantic map')
                    plt.axis('off')

                    # --- 核心修改：保存图片 ---
                    # 构造文件名：step_{时间步}_batch_{批次号}.png
                    # 假设外层循环变量是 stepk
                    file_name = f"step_{stepk:04d}_batch_{b:02d}.png"
                    save_path = os.path.join(save_dir, file_name)
                    
                    # 保存图片
                    plt.savefig(save_path, bbox_inches='tight', dpi=100)
                    # print(f"Saved visualization to {save_path}")
                    # print(f"batch_{stepk}_rel_abs_pose: {batch_rel_abs_pose[b].cpu().numpy().tolist()}")

                for b in range(batch_size):
                    waypoint_grid_maps[b] = waypoint_grid_maps[b] - waypoint_grid_maps[b].min()

                
                if observe_environment < 9:
                    self._send_rotate_observation_subgoal(
                        agent_state['position'],
                        observe_environment=observe_environment,
                        timestamp=ts  # 若你已有 ts，可直接传 ts
                    )
                    continue

                
                if self.world_subgoal is not None:
                    distance_to_subgoal = self.calculate_distance_xy(cur_location_world, self.world_subgoal)
                    if distance_to_subgoal > 0.6:
                        print(f"Step{stepk}, distance to subgoal: {distance_to_subgoal:.2f}m, skip planning.")
                        print(f"Step{stepk}, current Position:", agent_state['position'])
                        self._publish_subgoal_response(
                            subgoal=self.subgoal,
                            status="keep_subgoal",
                            timestamp=ts
                        )
                        continue
                    else:
                        print(f"Step{stepk}, distance to subgoal: {distance_to_subgoal:.2f}m, continue planning.")


                # waypoint_grid_maps = self.noise_filter(waypoint_grid_maps).squeeze(1)

                x = torch.arange(0, map_config['grid_dim'][0], dtype=torch.float16).to(self.device)
                y = torch.arange(0, map_config['grid_dim'][1], dtype=torch.float16).to(self.device)
                yg, xg = torch.meshgrid(y,x)
                yg = -(yg.to(self.device) -  map_config['grid_dim'][1] / 2. + 0.5)
                xg = xg.to(self.device) -  map_config['grid_dim'][0] / 2. + 0.5

                grid_rel_angle = torch.atan2(xg, yg)
                grid_rel_angle = (grid_rel_angle + 2*math.pi) % (2.*math.pi)

                predicted_waypoints = [[] for direction_idx in range(12)]

                for direction_idx in range(12):
                    back_angle = math.radians(direction_idx * 30.- 15.) 
                    front_angle = math.radians(direction_idx * 30.+ 15.)
                    if direction_idx == 0:
                        back_angle += 2.*math.pi
                        selected_part = (back_angle <= grid_rel_angle) | (grid_rel_angle <= front_angle)
                    else:
                        selected_part = (back_angle <= grid_rel_angle) & (grid_rel_angle <= front_angle)

                    tmp_waypoint_grid_maps = waypoint_grid_maps.clone()
                    tmp_waypoint_grid_maps[:,selected_part==False] = 0.
                    max_value, max_index = tmp_waypoint_grid_maps.view(batch_size,-1).max(dim=-1)
                    max_y = torch.div(max_index, map_config['grid_dim'][0], rounding_mode='floor')
                    max_x = max_index % map_config['grid_dim'][0]

                    predicted_waypoints[direction_idx] = torch.cat([max_value.view(batch_size,1),max_x.view(batch_size,1),max_y.view(batch_size,1)],dim=-1).unsqueeze(1)

                predicted_waypoints = torch.cat(predicted_waypoints,dim=1)

                # merge near waypoints
                merge_scale = 8
                for x_merge in range(2):
                    for y_merge in range(2):
                        tmp_predicted_waypoints = predicted_waypoints[:,:,1:].to(torch.int64)  
                        if x_merge == 1:
                            tmp_predicted_waypoints[:,:,0] = tmp_predicted_waypoints[:,:,0] + merge_scale
                        if y_merge == 1:
                            tmp_predicted_waypoints[:,:,1] = tmp_predicted_waypoints[:,:,1] + merge_scale

                        tmp_predicted_waypoints = torch.div(tmp_predicted_waypoints, merge_scale*2, rounding_mode='floor').to(torch.int32)
                        for b in range(batch_size):
                            tmp_dict = {}
                            for i in range(12):
                                # delete occupied waypoints
                                if predicted_occup_grid_maps[b,:,predicted_waypoints[b,i,1].to(torch.int64),predicted_waypoints[b,i,2].to(torch.int64)].argmax().cpu().item() == 1: # occupied
                                    predicted_waypoints[b,i,0] = 0.

                                key = str([tmp_predicted_waypoints[b][i][0].cpu().item(), tmp_predicted_waypoints[b][i][1].cpu().item()])
                                if key in tmp_dict:
                                    if predicted_waypoints[b,tmp_dict[key],0] > predicted_waypoints[b,i,0]:
                                        predicted_waypoints[b,i,0] = 0.
                                    else:
                                        predicted_waypoints[b,tmp_dict[key],0] = 0.
                                else:
                                    tmp_dict[key] = i


                # select k waypoints
                selected_waypoint_index = torch.topk(predicted_waypoints[:,:,0], k=3, dim=-1, largest=True)[1]
                selected_waypoints = [0 for b in range(batch_size)]
                batch_angle_idxes = []
                batch_distance_idxes = []
                for b in range(batch_size):
                    selected_waypoints[b] = predicted_waypoints[b,selected_waypoint_index[b]]
                    selected_waypoints[b] = selected_waypoints[b][selected_waypoints[b][:,0]!=0]
                    selected_waypoints[b] = selected_waypoints[b][:,1:]
                    rel_y = -(selected_waypoints[b][:,1] - map_config['grid_dim'][1]//2 + 0.5) * 0.05
                    rel_x = (selected_waypoints[b][:,0] - map_config['grid_dim'][0]//2 + 0.5) * 0.05
                    rel_angle = torch.atan2(rel_x, rel_y)

                    rel_dist = torch.sqrt(torch.square(rel_x) + torch.square(rel_y))
                    mask = (0.1 < rel_dist) & (rel_dist < 5.)
                    rel_dist = rel_dist[mask]
                    rel_angle = (rel_angle + 2*math.pi) % (2.*math.pi)
                    rel_angle = rel_angle[mask]
                    selected_waypoints[b] = selected_waypoints[b][mask]

                    # Discretization
                    angle_idx = torch.div((rel_angle+(math.pi/120)), (math.pi/60), rounding_mode='floor').to(torch.int32)
                    distance_idx = torch.div(rel_dist+0.25/2., 0.25, rounding_mode='floor').to(torch.int32) 

                    batch_angle_idxes.append(angle_idx)
                    batch_distance_idxes.append(distance_idx)


                
                for b in range(batch_size):            

                    plt.cla()
                    plt.imshow(waypoint_grid_maps[b].cpu().numpy())

                    fig = plt.gcf()
                    plt.margins(0,0)
                    fig.savefig("example/"+'heatmap_'+str(stepk)+'.png', dpi=500, bbox_inches='tight')
                    plt.show()

                    ################################## 
                    save_img_dir = "example/" + 'predicted_occup_' + str(stepk) +'_'
                    viz_utils.write_tensor_imgSegm(predicted_occup_grid_maps, save_img_dir, name="img_segm", t=stepk, labels=3)

                    save_img_dir = "example/" + 'predicted_semantic_' + str(stepk) +'_'
                    viz_utils.write_tensor_imgSegm(predicted_segm_grid_maps, save_img_dir, name="img_segm", t=stepk, labels=27)

                    save_img_dir = "example/" + 'occup_' + str(stepk) +'_'
                    viz_utils.write_tensor_imgSegm(step_occup_grid_maps, save_img_dir, name="img_segm", t=stepk, labels=3)


                    save_img_dir = "example/" + 'semantic_' + str(stepk) +'_'
                    viz_utils.write_tensor_imgSegm(step_segm_grid_maps, save_img_dir, name="img_segm", t=stepk, labels=27)


                    save_img_dir = "example/" + 'segm' + str(stepk) +'_'
                    viz_utils.write_tensor_imgSegm(img_segm, save_img_dir, name="img_segm", t=stepk, labels=27)


                    save_img_dir = "example/" + 'img_'
                    im_path = save_img_dir + str(stepk) + ".png"
                    Image.fromarray(viz_img).save(im_path)


                    save_img_dir = "example/" + 'depth_'
                    im_path = save_img_dir + str(stepk) + ".png"
                    Image.fromarray((viz_depth * 255).astype(np.uint8)).save(im_path)
                                       

                    save_img_dir = "example/" + 'predicted_cwp_' + str(stepk) +'_'
                    viz_utils.write_tensor_imgSegm(predicted_segm_grid_maps, save_img_dir, name="img_segm", t=stepk, labels=27, waypoints=selected_waypoints[b].cpu().to(torch.int64).numpy().tolist())

                
                # cand waypoint representation, need to be freezed
                wp_outputs = self.policy.net(
                    mode = "semantic_map_for_waypoint",
                    batch_angle_idxes = batch_angle_idxes,
                    batch_distance_idxes = batch_distance_idxes,
                    observations = observations,
                    in_train = False,
                )
                    
            ###############################
                
            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_node
            cur_pos, cur_ori = positions, headings
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.batch_size):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)


            cur_path_ghost = []
            for i in range(self.batch_size):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]
                ghost_set = self.gmaps[i].update_graph(prev_vp[i], prev_ghost[i], prev_ghost_emb[i], stepk+1,
                                            cur_vp[i], cur_pos[i], cur_embeds,
                                            cand_vp[i], cand_pos[i], cand_embeds,)
                cur_path_ghost.append(ghost_set)

            total_actions += self.batch_size
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]
                
            MAM_inputs = self.get_MAM_variable(cur_vp, cur_path_ghost, txt_embeds, txt_masks)
            MAM_inputs.update({
                'mode': 'MAM'
            })
            path_tokens = self.policy.net(**MAM_inputs)
            for ei in range(self.batch_size):
                self.gmaps[ei].update_path_embeds(path_tokens[ei], cur_vp[ei], cur_path_ghost[ei])
                
            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            # ###################################
            # nav_inputs['predicted_bev_fts'] = wp_outputs['predicted_bev_fts']
            # ###################################
            nav_outs = self.policy.net(**nav_inputs)
            global_logits = nav_outs['global_logits']

            CCM_inputs = self.get_CCM_variable(cur_vp)
            CCM_inputs.update({
                    'mode': 'CCM',
                    'gmap_features': nav_outs['gmap_embeds'],
                    'gmap_logits': global_logits,
                    'gmap_vp_ids': nav_inputs['gmap_vp_ids'],
                    'txt_embeds': txt_embeds,
                    'txt_masks': ~txt_masks,
                })
            path_logits, global_logits, residual_logits, final_logits = self.policy.net(**CCM_inputs)

            final_logits = final_logits

            nav_probs = F.softmax(final_logits, dim=-1)
            path_probs = F.softmax(path_logits, dim=-1)
            residual_probs = F.softmax(residual_logits, dim=-1)
            global_probs = F.softmax(global_logits, dim=-1)

            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()
            

            a_t = final_logits.argmax(dim=-1)

            cpu_a_t = a_t.cpu().numpy()
          
            # make equiv action
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    print("Stop.")

                    self.subgoal = self.pack_action(agent_state['position'],agent_state['position'])
                    # FastAPI send original Subgoal
                    self._publish_subgoal_response(
                        subgoal=self.subgoal,
                        status="stop",
                        timestamp=ts
                    )

                    print(batch_rel_abs_pose)
                    exit()
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)

                    prev_vp[i] = front_vp
                    prev_ghost[i] = ghost_vp
                    prev_ghost_emb[i] = gmap.get_ghost_front_edges(ghost_vp, front_vp)
                    if self.config['MODEL']['consume_ghost']:
                        gmap.delete_ghost(ghost_vp)

                    print(f"Current position:{agent_state['position']}")
                    print(f"Planning Step {total_actions} Subgoal: {ghost_pos.tolist()}.")
                    self.subgoal = self.pack_action(agent_state['position'],ghost_pos)

                    self.world_subgoal = np.array(self.subgoal['position'][1])

                    # FastAPI send new Subgoal

                    self._publish_subgoal_response(
                        subgoal=self.subgoal,
                        status="new_subgoal",
                        timestamp=ts
                    )