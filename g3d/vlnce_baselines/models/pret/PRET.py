import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM, CLIPModel, CLIPProcessor, BertModel

from vlnce_baselines.pret_utils.mask import get_causal_mask
from vlnce_baselines.models.etp.vilmodel_cmt import BertIntermediate, BertOutput

config = BertModel.config_class.from_pretrained("/home/houjiewen/g3D-LF-main/Lookahead_3DFF/bert_config/bert-base-uncased")

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
            
        # 建立網絡層
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            BertLayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
        # 應用權重初始化
        self.apply(self.init_weights)

    def init_weights(self, module):
        """
        初始化權重。此函數會被 self.apply() 遞迴調用。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用常態分佈初始化 Linear 和 Embedding 層的權重
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            # 將 LayerNorm 的 bias 設為 0，weight 設為 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
        # 如果是 Linear 層且有 bias，則將 bias 設為 0
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        return self.net(x)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs

class X_Layer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):      
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

class FusionProjector(nn.Module):
    """
    一個合理的 MLP Projector，用於融合拼接後的特徵。
    採用 "線性 -> 激活 -> 線性" 的瓶頸結構。
    """
    def __init__(self, in_dim, out_dim, hidden_dim_ratio=2.0, dropout=0.1):
        super().__init__()
        # 隱藏層的維度，可以根據需求調整
        hidden_dim = int(in_dim * hidden_dim_ratio)
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),  # GELU 是比 ReLU 更平滑、現代的激活函數，在 Transformer 中常用
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.projector(x)

def extend_masks(masks, dtype=None):
    """
    mask from (N, L) into (N, 1(H), 1(L), L) and make it negative
    """
    if dtype is None:
        dtype = torch.float
    extended_masks = masks.unsqueeze(1).unsqueeze(2)
    extended_masks = extended_masks.to(dtype=dtype)
    extended_masks = extended_masks * -10000.0
    return extended_masks

def extend_neg_masks(masks, dtype=None):
    """
    mask from (N, L) into (N, 1(H), 1(L), L) and make it negative
    """
    if dtype is None:
        dtype = torch.float
    extended_masks = masks.unsqueeze(1).unsqueeze(2)
    extended_masks = extended_masks.to(dtype=dtype)
    extended_masks = (1.0 - extended_masks) * -10000.0
    return extended_masks


class PRET(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.multilingual = False
        self.device = device
        # self.dropout = 0.5
        self.hidden_dim = 768
        self.nhead = 12
        # self.OPE_layer_num = 2
        self.MAM_layer_num = 4
        self.CCM_layer_num = 1

        self.fuse_layer_num = 1

        # self.dropout = nn.Dropout(self.dropout)

        # OPE
        # It is strange that dropout in transformer is not reproducible even with fixed seed
        # I use pytorch 1.13 and cuda 11.8
        # self.angle_embedding = nn.Linear(4, self.hidden_dim)
        # self.vision_embedding = nn.Linear(512, self.hidden_dim)

        self.use_panorama = True
        self.use_directed = True  # if not use panorama, this is ignored
        # if self.use_panorama:
        #     if self.use_directed:
        #         self.OPE = get_transformer_decoder(self.hidden_dim, self.nhead, self.OPE_layer_num)
        #     else:
        #         self.OPE = get_transformer_encoder(self.hidden_dim, self.nhead, self.OPE_layer_num)

        # MAM
        self.MAM = get_transformer_decoder(self.hidden_dim, self.nhead, self.MAM_layer_num)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # CCM
        self.CCM = get_transformer_encoder(self.hidden_dim, self.nhead, self.CCM_layer_num)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fuse_self_att = BertAttention(config)
        self.cross_modal_fuse = X_Layer(config)
        self.fuse_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.norm = BertLayerNorm(self.hidden_dim, eps=1e-12)

        self.gate = ClsPrediction(self.hidden_dim, input_size=2 * self.hidden_dim)


        for param in self.CCM.parameters():
            param.requires_grad = False
        self.cls.requires_grad = False
        for param in self.MAM.parameters():
            param.requires_grad = False
        # for param in self.mlp.parameters():
        #     param.requires_grad = False
        

    def forward_MAM(self,
            text_features,
            text_padding_mask,
            path_features,
            path_padding_mask,
            local_features=None,
            local_padding_mask=None,):
        B, N, C = path_features.shape
        path_only = local_features is None
        if path_only:
            local_features = torch.zeros(B, 0, C, device=path_features.device)
            local_padding_mask = torch.zeros(B, 0, dtype=torch.bool, device=path_features.device)

        assert text_padding_mask.dtype == torch.bool
        assert path_padding_mask.dtype == torch.bool
        assert local_padding_mask.dtype == torch.bool

        B, M, C = local_features.shape
        pos = torch.arange(N, device=path_features.device)
        pe = position_embedding(pos, dim=C)
        path_features = path_features + pe

        pe = position_embedding(torch.tensor([N], device=local_features.device), dim=C)
        local_features[:, 1:, :] = local_features[:, 1:, :] + pe  # add position embedding except STOP

        # prepare inputs, cls is used as start token
        cls = self.cls.expand(B, 1, -1)
        falses = torch.zeros(B, cls.shape[1], dtype=torch.bool, device=cls.device)
        tokens = torch.cat([cls, path_features, local_features], dim=1)
        padding_mask = torch.cat([falses, path_padding_mask, local_padding_mask], dim=1)

        # prepare merged causal mask
        mask = torch.zeros(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=self.device)
        i = N + cls.shape[1]  # +1 for cls
        j = i + M
        mask[i:, :] = True
        mask[:i, :i] = get_causal_mask(i).to(tokens.device)
        mask[i:j, i:j] = torch.eye(local_features.shape[1], dtype=torch.bool, device=self.device)
        mask = ~mask

        # forward
        tokens = self.MAM(
            tokens, text_features,
            tgt_mask=mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=text_padding_mask)

        # return
        cls, path_tokens, local_tokens = torch.split(tokens, [cls.shape[1], N, M], dim=1)
        if path_only:
            return path_tokens
        else:
            return path_tokens, local_tokens

    def forward_CCM(self, path_features, gmap_features, gmap_logits, txt_embeds, txt_masks, padding_mask=None):
        """
        Args:
            path_feature: (B, N, C)
            padding_mask: (B, N)
        Returns:
            (B, N)
        """
        
        path_features = self.CCM(path_features, src_key_padding_mask=padding_mask)
        path_logits = self.mlp(path_features).squeeze(2)
        weight = torch.sigmoid(self.gate(torch.cat([path_features[:, 0], gmap_features[:, 0]], dim=-1)))
        path_features = path_features.detach()  # detach to avoid gradient flow to CCM
        gmap_features = gmap_features.detach()  # detach to avoid gradient flow to CCM
        weight = weight.unsqueeze(1)

        B, N, C = path_features.shape
        # pos_p = position_embedding(torch.arange(N, device=path_features.device))
        # pos_g = position_embedding(torch.arange(N, device=gmap_features.device))
        # path_features = path_features + pos_p
        # gmap_features = gmap_features + pos_g
        vis_mask = extend_masks(padding_mask)
        txt_mask = extend_masks(txt_masks)

        fuse_feat = self.norm(weight * path_features + (1 - weight) * gmap_features)

        fuse_feat = self.cross_modal_fuse(txt_embeds, txt_mask, fuse_feat, vis_mask)
        fuse_feat = self.fuse_self_att(fuse_feat, vis_mask)[0]
        residual_logits = self.fuse_head(fuse_feat).squeeze(2)


        gmap_logits = gmap_logits - gmap_logits.max(dim=1, keepdim=True).values
        path_logits = path_logits - path_logits.max(dim=1, keepdim=True).values
        residual_logits = residual_logits - residual_logits.max(dim=1, keepdim=True).values

        gmap_logits.masked_fill_(padding_mask, -float('inf'))
        path_logits.masked_fill_(padding_mask, -float('inf'))
        residual_logits.masked_fill_(padding_mask, -float('inf'))
        final_logits = 1.0 * (path_logits +  gmap_logits) + residual_logits

        return path_logits, gmap_logits, residual_logits, final_logits, weight


def get_transformer_encoder(hidden_dim, nhead, layer_num):
    encoder_layer = nn.TransformerEncoderLayer(
        hidden_dim,
        nhead=nhead,
        dim_feedforward=hidden_dim * 4,
        activation=F.gelu,
        batch_first=True)
    return nn.TransformerEncoder(encoder_layer, num_layers=layer_num, enable_nested_tensor=False)


def get_transformer_decoder(hidden_dim, nhead, layer_num):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=nhead,
        dim_feedforward=hidden_dim * 4,
        activation=F.gelu,
        batch_first=True
    )
    return nn.TransformerDecoder(decoder_layer, num_layers=layer_num)


def position_embedding(pos, dim=768, max_len=10000):
    """
    Args:
        pos: tensor
        dim: dimension of the token, normally it is 768.
        device: create the tensor on cpu or gpu.
    Returns:
        position embedding, a (dim,) tensor
    """
    x = torch.arange(1, dim + 1, device=pos.device)
    phi = (x % 2 == 0) * (torch.pi / 2)

    x[x % 2 == 1] += 1
    if isinstance(pos, torch.Tensor):
        for i in range(len(pos.shape)):
            x.unsqueeze(0)
        x = pos.unsqueeze(-1) / (max_len ** (x / dim))
    else:
        x = pos / (max_len ** (x / dim))

    pe = torch.sin(x + phi)
    return pe
