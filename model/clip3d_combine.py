import math
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip.vision_transformer_LoRA import vit_base_patch16_224, vit_large_patch14_clip_224
import numpy as np
from torch.nn import init
from .transmil import TransMIL
from .attnmil import Attn_Net_Gated


class BatchedTextConditionedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, query_embeds, video_embeds, query_mask=None):
        """
        query_embeds: [B, Q, C]
        video_embeds: [B, T, C]
        query_mask: [B, Q] bool, True means valid query
        """
        attn_logits = torch.matmul(query_embeds, video_embeds.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        attended = torch.matmul(attn_weights, video_embeds)

        if query_mask is not None:
            mask = query_mask.unsqueeze(-1).to(attended.dtype)
            attended = attended * mask
            attn_weights = attn_weights * mask

        return attended, attn_weights

class MultiHeadedAttention_xpool(nn.Module):
    def __init__(self, embed_dim, num_mha_heads):
        super(MultiHeadedAttention_xpool, self).__init__()
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
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

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

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
   
        return o, attention_weights

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
    def __init__(self, embed_dim, num_mha_heads, transformer_dropout, using_xpooling):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        dropout = transformer_dropout

        if using_xpooling:
            self.cross_attn = MultiHeadedAttention_xpool(embed_dim, num_mha_heads)
        else:
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
        # out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(linear_out)

        return out, attn_weights


class CLIPAdapter(nn.Module):
    def __init__(self, c_in, reduction=4, ratio=0.2):
        super(CLIPAdapter, self).__init__()
        hidden_dim = max(c_in // reduction, 1)
        self.ratio = ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        adapted = self.fc(x)
        return self.ratio * adapted + (1 - self.ratio) * x

class CLIPTransformer(nn.Module):
    def __init__(self, clip_backbone, freeze_backbone, freeze_text_backbone, embed_dim=512, num_mha_heads=1, transformer_dropout=0.3, 
        backbone_name='biomedclip', backbone_ckpt='', prompt_length=0, use_vpt=False, vision_ctx=0, use_lora=False, use_clip_adapter=False, clip_adapter_reduction=4, clip_adapter_ratio=0.2,
        lora_rank=4, lora_alpha=1, out_cls=2, add_reg_head=True, add_cls_head=False,add_retrieval_reg_head=False, ct_mil=False, 
        using_xpooling=False, using_attnmil=False, add_multi_cls=False,
        paper_core_mode=False, max_fine_grained_sentences=8, use_uar=False, uar_alpha=2.0, uar_lambda=0.5):
        super(CLIPTransformer, self).__init__()
        self.clip_backbone = clip_backbone
        self.backbone_name = backbone_name.lower()
        self.supports_peft_injection = hasattr(self.clip_backbone, "visual") and hasattr(self.clip_backbone.visual, "trunk")
        self.backbone_output_dim = getattr(self.clip_backbone.visual, 'output_dim', 512)
        if self.supports_peft_injection:
            vit = vit_base_patch16_224(False, prompt_depth=prompt_length, use_vpt=use_vpt, vision_ctx=vision_ctx, 
                    use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha )
            vit.reset_classifier(0)
            try:
                vit.load_state_dict(self.clip_backbone.visual.trunk.state_dict())
            except:
                missing_keys, _ = vit.load_state_dict(self.clip_backbone.visual.trunk.state_dict(), strict=False)
                # print('Weights not found for some missing keys: ', missing_keys)
            self.clip_backbone.visual.trunk = vit
        elif self.backbone_name == 'radclip' and use_vpt:
            vit = vit_large_patch14_clip_224(False, prompt_depth=prompt_length, use_vpt=use_vpt, vision_ctx=vision_ctx)
            vit.reset_classifier(0)
            rad_vit_state = self._convert_hf_clip_vision_to_timm_vit(self.clip_backbone.vision_model.state_dict())
            missing_keys, _ = vit.load_state_dict(rad_vit_state, strict=False)
            self.clip_backbone.vision_model = HFVisionModelWrapper(vit)
        self.use_clip_adapter = use_clip_adapter
        if self.use_clip_adapter:
            self.clip_adapter = CLIPAdapter(
                c_in=self.backbone_output_dim,
                reduction=clip_adapter_reduction,
                ratio=clip_adapter_ratio,
            )
        if freeze_backbone:
            for name, param in self.clip_backbone.named_parameters():
                if self.supports_peft_injection and "VPT" in name and use_vpt:
                    param.requires_grad_(True)
                elif self.supports_peft_injection and "lora" in name and use_lora:
                    param.requires_grad_(True)
                elif "logit_scale" in name:
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
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = self.clip_backbone.logit_scale
        if self.backbone_output_dim != 512:
            self.visual_dim_proj = nn.Linear(self.backbone_output_dim, 512)
            self.text_dim_proj = nn.Linear(self.backbone_output_dim, 512)
        else:
            self.visual_dim_proj = nn.Identity()
            self.text_dim_proj = nn.Identity()
        if using_attnmil:
            self.pool_slices = Attn_Net_Gated(L=512, D=256, dropout=False, n_classes=1)
        else:
            self.pool_slices = Transformer(embed_dim, num_mha_heads, transformer_dropout, using_xpooling)
        
        # self.fusion_layer = nn.Linear(1024, 512)
        self.add_cls_head = add_cls_head
        self.add_reg_head = add_reg_head
        self.add_retrieval_reg_head = add_retrieval_reg_head
        self.add_multi_cls = add_multi_cls
        self.ct_mil = ct_mil
        self.using_xpooling = using_xpooling
        self.using_attnmil = using_attnmil
        self.paper_core_mode = paper_core_mode
        self.max_fine_grained_sentences = max_fine_grained_sentences
        self.use_uar = use_uar
        self.uar_alpha = uar_alpha
        self.uar_lambda = uar_lambda

        if self.paper_core_mode:
            self.tca = BatchedTextConditionedAttention(512)
            self.logic_pool = Attn_Net_Gated(L=512, D=256, dropout=False, n_classes=1)

        if self.add_cls_head:
            self.cls_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, out_cls))
        if self.add_reg_head:
            self.reg_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
        if self.add_retrieval_reg_head:
            self.retrieval_reg_head = SSRModule()
        if self.ct_mil:
            self.mil_head = TransMIL(n_classes=1,embed_dim=512,dropout=False,act='relu')
        if self.add_multi_cls:
            self.multi_cls_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 21))
        # self.tail = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, out_cls))
        # self.tail = nn.Linear(512, 1)


    def forward(self, texts, img):
        logits = {}
        bs = img.shape[0]
        num_slices = img.shape[2]
        img = img.transpose(1, 2)
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]) #[B*T,3,H,W]

        img_embeds = self.clip_backbone.encode_image(img)  #[B*T,E]
        if self.use_clip_adapter:
            img_embeds = self.clip_adapter(img_embeds)
        img_embeds = self.visual_dim_proj(img_embeds)
        img_embeds = img_embeds.reshape(bs, num_slices, -1) #[B,T,E]

        if self.paper_core_mode:
            text_embeds_all = self.encode_projected_text(texts['text_all'])
            fine_grained_embeds = self._encode_grouped_text(texts.get('text_fine_grained'))
            fine_grained_mask = texts.get('text_fine_grained_mask')

            global_tca_features, global_tca_attn = self.tca(text_embeds_all.unsqueeze(1), img_embeds)
            global_tca_features = global_tca_features.squeeze(1)

            if fine_grained_embeds is not None:
                fine_visual_features, fine_tca_attn = self.tca(
                    fine_grained_embeds, img_embeds, fine_grained_mask
                )
            else:
                fine_visual_features, fine_tca_attn = None, None

            logic_features, logic_attn = self.logic_pool(img_embeds)
            logic_features = logic_features.squeeze(1)

            if self.add_multi_cls:
                logits['multi_cls'] = self.multi_cls_head(logic_features)

            logits['paper_aux'] = {
                'logic_features': logic_features,
                'logic_attn': logic_attn,
                'global_tca_features': global_tca_features,
                'global_tca_attn': global_tca_attn,
                'global_text_features': text_embeds_all,
                'fine_visual_features': fine_visual_features,
                'fine_text_features': fine_grained_embeds,
                'fine_tca_attn': fine_tca_attn,
                'fine_grained_mask': fine_grained_mask,
                'slice_features': img_embeds,
            }

            return logits, logic_features, logic_attn, text_embeds_all, fine_grained_embeds, img_embeds


        if not self.using_attnmil:
            text_embeds_all = self.encode_projected_text(texts['text_all']) #[B,E]
            text_embeds_categorize = self.encode_projected_text(texts['text_categorize']) #[L,E]
            # text_embeds_basic = self.clip_backbone.encode_text(text_basic) #[B,E]
            # text_embeds_ct = self.clip_backbone.encode_text(text_ct) #[B,E]
            
            # text_embeds_unCT = text_embeds[:bs]
            # text_embeds_CT = text_embeds[bs:]
            out_aggregated, attn_weights = self.pool_slices(text_embeds_all, img_embeds)
            if self.using_xpooling:
                out = torch.diagonal(out_aggregated,dim1=0,dim2=1).t() + text_embeds_all  #[B,B,E]->[B,E]
            else:
                out = out_aggregated.squeeze(1) #[B,E]
                # out = out + text_embeds_all
                # out = self.fusion_layer(torch.cat([out_aggregated, text_embeds_basic], dim=-1))
        else:
            out_aggregated, attn_weights = self.pool_slices(img_embeds)
            out = out_aggregated.squeeze(1) #[B,E]
            text_embeds_all, text_embeds_categorize = None, None
            # out = out + text_embeds_all
            # out = self.fusion_layer(torch.cat([out_aggregated, text_embeds_basic], dim=-1))

        if self.add_reg_head:
            logits_reg = self.reg_head(out)
            logits['reg'] = logits_reg
        if self.add_cls_head:
            logits_cls = self.cls_head(out)
            logits['cls'] = logits_cls
        if self.add_retrieval_reg_head:
            text_embeds_categorize = text_embeds_categorize / text_embeds_categorize.norm(dim=-1, keepdim=True)
            sims = (out / out.norm(dim=-1, keepdim=True)) @ text_embeds_categorize.t() #[B,L]
            sims = sims * self.logit_scale.exp()
            logits_retrieval_reg = self.retrieval_reg_head(sims)
            logits['reg'] = logits_retrieval_reg
            logits['sims_retrieval'] = sims
        if self.ct_mil:
            logits_reg_ct_mil = self.mil_head(img_embeds)
            logits['reg_ct_mil'] = logits_reg_ct_mil
        if self.add_multi_cls:
            logits_multi_cls = self.multi_cls_head(out)
            logits['multi_cls'] = logits_multi_cls

        return logits, out_aggregated, attn_weights, text_embeds_all, text_embeds_categorize, out #img_embeds

    def encode_projected_text(self, text_tokens):
        if text_tokens is None:
            return None
        text_embeds = self.clip_backbone.encode_text(text_tokens)
        return self.text_dim_proj(text_embeds)

    def _encode_grouped_text(self, grouped_tokens):
        if grouped_tokens is None:
            return None

        if isinstance(grouped_tokens, dict):
            first_tensor = next(iter(grouped_tokens.values()))
            batch_size, group_size = first_tensor.shape[:2]
            flattened = {
                key: value.reshape(batch_size * group_size, *value.shape[2:])
                for key, value in grouped_tokens.items()
            }
        else:
            batch_size, group_size = grouped_tokens.shape[:2]
            flattened = grouped_tokens.reshape(batch_size * group_size, *grouped_tokens.shape[2:])

        text_embeds = self.encode_projected_text(flattened)
        return text_embeds.reshape(batch_size, group_size, -1)

    def compute_prompt_semantic_scores(self, slice_features, class_text_embeds, temperature=0.07):
        batch_size = slice_features.shape[0]
        query_embeds = class_text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        class_visual_features, _ = self.tca(query_embeds, slice_features)
        class_visual_features = F.normalize(class_visual_features, dim=-1)
        class_text_embeds = F.normalize(class_text_embeds, dim=-1).unsqueeze(0)
        cosine_scores = (class_visual_features * class_text_embeds).sum(dim=-1)
        return torch.sigmoid(cosine_scores / temperature)

    def _convert_hf_clip_vision_to_timm_vit(self, hf_state_dict):
        timm_state = {}
        timm_state["patch_embed.proj.weight"] = hf_state_dict["embeddings.patch_embedding.weight"]
        if "embeddings.patch_embedding.bias" in hf_state_dict:
            timm_state["patch_embed.proj.bias"] = hf_state_dict["embeddings.patch_embedding.bias"]
        timm_state["cls_token"] = hf_state_dict["embeddings.class_embedding"].reshape(1, 1, -1)
        timm_state["pos_embed"] = hf_state_dict["embeddings.position_embedding.weight"].unsqueeze(0)
        timm_state["norm_pre.weight"] = hf_state_dict["pre_layrnorm.weight"]
        timm_state["norm_pre.bias"] = hf_state_dict["pre_layrnorm.bias"]
        timm_state["norm.weight"] = hf_state_dict["post_layernorm.weight"]
        timm_state["norm.bias"] = hf_state_dict["post_layernorm.bias"]

        block_ids = sorted({
            int(k.split(".")[2])
            for k in hf_state_dict.keys()
            if k.startswith("encoder.layers.")
        })
        for idx in block_ids:
            hf_prefix = f"encoder.layers.{idx}"
            timm_prefix = f"blocks.{idx}"

            q_w = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"]
            k_w = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"]
            v_w = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"]
            q_b = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.bias"]
            k_b = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.bias"]
            v_b = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.bias"]

            timm_state[f"{timm_prefix}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            timm_state[f"{timm_prefix}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
            timm_state[f"{timm_prefix}.attn.proj.weight"] = hf_state_dict[f"{hf_prefix}.self_attn.out_proj.weight"]
            timm_state[f"{timm_prefix}.attn.proj.bias"] = hf_state_dict[f"{hf_prefix}.self_attn.out_proj.bias"]
            timm_state[f"{timm_prefix}.norm1.weight"] = hf_state_dict[f"{hf_prefix}.layer_norm1.weight"]
            timm_state[f"{timm_prefix}.norm1.bias"] = hf_state_dict[f"{hf_prefix}.layer_norm1.bias"]
            timm_state[f"{timm_prefix}.norm2.weight"] = hf_state_dict[f"{hf_prefix}.layer_norm2.weight"]
            timm_state[f"{timm_prefix}.norm2.bias"] = hf_state_dict[f"{hf_prefix}.layer_norm2.bias"]
            timm_state[f"{timm_prefix}.mlp.fc1.weight"] = hf_state_dict[f"{hf_prefix}.mlp.fc1.weight"]
            timm_state[f"{timm_prefix}.mlp.fc1.bias"] = hf_state_dict[f"{hf_prefix}.mlp.fc1.bias"]
            timm_state[f"{timm_prefix}.mlp.fc2.weight"] = hf_state_dict[f"{hf_prefix}.mlp.fc2.weight"]
            timm_state[f"{timm_prefix}.mlp.fc2.bias"] = hf_state_dict[f"{hf_prefix}.mlp.fc2.bias"]

        return timm_state
        # logits = {}
        # img = img.transpose(1, 2) #[B,T,3,H,W]
        # img_embeds = self.clip_backbone.encode_image(img[:,18])
        # out = img_embeds

        # # out = self.fusion_layer(torch.cat([out_aggregated, text_embeds_basic], dim=-1))
        # if self.add_reg_head:
        #     logits_reg = self.reg_head(out)
        #     logits['reg'] = logits_reg
        # if self.add_cls_head:
        #     logits_cls = self.cls_head(out)
        #     logits['cls'] = logits_cls

        # out, attn_weights, text_embeds_all = None, None, None
        # return logits, out, attn_weights, text_embeds_all, img_embeds


class SSRModule(nn.Module):
    def __init__(self, d=512,
                 class_range=101, lambda_index=1., lambda_delta=1.):
        super(SSRModule, self).__init__()

        self.bin_list = [7, 15, 20, 40] 
        self.stage_num = len(self.bin_list)
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range
        self.d = d
        

        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num, 2 * self.stage_num),
            nn.ReLU()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(2 * self.stage_num, self.stage_num),
            nn.Tanh()
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    
    def forward(self, logits):
        '''
        logits: [B,L], L means the nums of split range 
        return: [B]
        '''
        prob_stage_1 = F.softmax(logits, dim=1) # [B,L]
        embedding_stage1_after_PB = self.stage1_FC_after_PB(logits)  
        stage1_delta_k = self.stage1_delta_k(embedding_stage1_after_PB) #[B,stage_num]

        stage1_regress_a = prob_stage_1[:, 0] * 0 # [B]

        for index in range(self.stage_num):
            width = (self.bin_list[index] / (1 + self.lambda_delta * stage1_delta_k[:, index]))
            stage1_regress_a = stage1_regress_a + prob_stage_1[:, index] * width
        # stage1_regress_a = torch.unsqueeze(stage1_regress_a, 1)

        # regress_age_a = stage1_regress_a
        # regress_age_a = regress_age_a.squeeze(1)

        # regress_age = regress_age_a

        return stage1_regress_a


class HFVisionOutput(SimpleNamespace):
    pass


class HFVisionModelWrapper(nn.Module):
    def __init__(self, timm_vit):
        super().__init__()
        self.timm_vit = timm_vit

    def forward(self, pixel_values):
        pooled = self.timm_vit(pixel_values)
        return HFVisionOutput(pooler_output=pooled)
