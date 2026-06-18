import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip.vision_transformer_LoRA import vit_base_patch16_224
import numpy as np

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_mha_heads):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        # num_texts x num_heads x head_dim x 1
        q = q.reshape(num_texts, self.num_heads, self.head_dim, 1) #[num_texts x num_heads x head_dim x 1]
        # num_heads x head_dim x num_texts
        # q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x 1
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x embed_dim x 1
        attention = v @ attention_weights
        # num_vids x 1 x num_heads x embed_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, 1, self.embed_dim)

        # num_vids x 1 x embed_dim  
        o = self.out_proj(attention)
        return o, attention_weights
    

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_mha_heads, transformer_dropout):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        dropout = transformer_dropout

        self.cross_attn = MultiHeadedAttention(embed_dim, num_mha_heads)

        self.linear_proj_in_text = nn.Linear(512, self.embed_dim)
        self.linear_proj_in_video = nn.Linear(512, self.embed_dim)
        self.linear_proj = nn.Linear(self.embed_dim, 512)
            
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(512)
        # self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        text_embeds = self.linear_proj_in_text(text_embeds)
        video_embeds = self.linear_proj_in_video(video_embeds)

        text_embeds = self.layer_norm2(text_embeds)
        video_embeds = self.layer_norm2(video_embeds)

        # num_vids x [1 or num_texts] x embed_dim
        attn_out, attn_weights = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        # out = attn_out + linear_out
        out = self.layer_norm3(linear_out)

        return out, attn_weights

class CLIPTransformer(nn.Module):
    def __init__(self, clip_backbone, freeze_backbone, freeze_text_backbone, embed_dim=512, num_mha_heads=1, transformer_dropout=0.3, 
        prompt_length=0, use_vpt=False, vision_ctx=0, use_lora=False, lora_rank=4, lora_alpha=1, out_cls=2,add_reg_head=True, add_cls_head=False):
        super(CLIPTransformer, self).__init__()
        self.clip_backbone = clip_backbone
        vit = vit_base_patch16_224(False, prompt_depth=prompt_length, use_vpt=use_vpt, vision_ctx=vision_ctx, 
                use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha )
        vit.reset_classifier(0)
        try:
            vit.load_state_dict(self.clip_backbone.visual.trunk.state_dict())
        except:
            missing_keys, _ = vit.load_state_dict(self.clip_backbone.visual.trunk.state_dict(), strict=False)
            # print('Weights not found for some missing keys: ', missing_keys)
        self.clip_backbone.visual.trunk = vit
        if freeze_backbone:
            for name, param in self.clip_backbone.named_parameters():
                if "VPT" in name and use_vpt:
                    param.requires_grad_(True)
                elif "lora" in name and use_lora:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        else:
            if freeze_text_backbone:
                for name, param in self.clip_backbone.named_parameters():
                    if "text" in name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pool_slices = Transformer(embed_dim, num_mha_heads, transformer_dropout)
        
        # self.fusion_layer = nn.Linear(1024, 512)
        self.tail = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, out_cls))
        # self.tail = nn.Linear(512, 1)
    

    def forward(self, text_basic, text_ct, img):
        bs = img.shape[0]
        num_slices = img.shape[2]
        img = img.transpose(1, 2)
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]) #[B*T,3,H,W]
        text_embeds_CT = self.clip_backbone.encode_text(text_ct) #[B,E]
        text_embeds_basic = self.clip_backbone.encode_text(text_basic) #[B,E]
        # text_embeds_unCT = text_embeds[:bs]
        # text_embeds_CT = text_embeds[bs:]

        img_embeds = self.clip_backbone.encode_image(img)  #[B,T,E1,E2]
        img_embeds = img_embeds.reshape(bs, num_slices, -1) #[B,T,E]
        out_aggregated, attn_weights = self.pool_slices(text_embeds_CT, img_embeds)

        out_aggregated = out_aggregated.squeeze(1) #[B,E]
        out = out_aggregated
        # out = out_aggregated + text_embeds_basic
        # out = self.fusion_layer(torch.cat([out_aggregated, text_embeds_basic], dim=-1))
        
        out = self.tail(out)
        
        return out, out_aggregated, attn_weights, text_embeds_CT, img_embeds