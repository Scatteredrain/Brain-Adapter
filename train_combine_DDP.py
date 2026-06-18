# train_ddp.py
import os
import random
import logging
import re
from datetime import datetime
import builtins
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import inspect

# ====== 你的项目依赖 ======
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from configs import default_argument_parser
from utils.get_util import (
    plot_and_save_scatter, sim_matrix, resize, separate_clip_params, log_losses,
    AvgMeter, log_metrics, evaluate_predictions, check_config,
    generate_categorize_text, save_training_data, gather_features_with_grad, YScaler,
    log_lrs
)
from utils.data_utils_5fold import get_loader
from utils.criterion import (
    ContrastiveLoss, RPS, sigmoid_focal_loss, CLIPLoss, Retrieval_Regression_Loss,
    BMCLoss, CLIP_dis_loss, categorical_ordinal_focal_loss, weighted_mse_loss, spearman_rank_loss, AsymmetricLossOptimized,
    huber_ccc_loss, HuberCCCLoss, FocalLoss
)
from model.clip3d_combine import CLIPTransformer
from model.backbone_loader import build_backbone


def uses_regression_path(config):
    return config.model.add_reg_head or config.model.add_retrieval_reg_head or config.model.ct_mil


def uses_multilabel_path(config):
    return config.model.add_multi_cls and config.loss.lambda_multi_cls > 0


def uses_paper_core_mode(config):
    return getattr(config.model, "paper_core_mode", False)


def uses_clip_alignment_path(config):
    return config.loss.lambda_clip > 0


def get_base_model(model):
    return model.module if isinstance(model, DDP) else model


def needs_scalar_targets(config):
    return uses_regression_path(config) or uses_clip_alignment_path(config)


def needs_categorize_text(config):
    return (not config.model.using_attnmil) and not uses_paper_core_mode(config)


def uses_cls_targets(config):
    return config.model.add_cls_head or config.model.add_retrieval_reg_head


def uses_onehot_targets(config):
    return config.loss.lambda_categorical_ordinal_focal_loss > 0 and config.model.add_cls_head


def tracks_scalar_predictions(config):
    return uses_regression_path(config)


def checkpoint_metric_mode(config):
    if uses_regression_path(config):
        return "regression"
    if config.model.add_cls_head:
        return "classification"
    if uses_multilabel_path(config):
        return "multilabel"
    if uses_clip_alignment_path(config):
        return "clip"
    return "none"


def format_active_paths(config):
    return (
        f"paper_core={uses_paper_core_mode(config)}, "
        f"multilabel={uses_multilabel_path(config)}, "
        f"regression={uses_regression_path(config)}, "
        f"clip_alignment={uses_clip_alignment_path(config)}, "
        f"needs_scalar_targets={needs_scalar_targets(config)}"
    )


def validate_runtime_config(config):
    errors = []

    if uses_paper_core_mode(config):
        if not config.model.add_multi_cls:
            errors.append("paper_core_mode requires model.add_multi_cls=True.")
        if config.loss.lambda_multi_cls <= 0:
            errors.append("paper_core_mode requires loss.lambda_multi_cls > 0.")
        if config.model.add_cls_head:
            errors.append("paper_core_mode is incompatible with model.add_cls_head.")
        if config.model.add_reg_head:
            errors.append("paper_core_mode is incompatible with model.add_reg_head.")
        if config.model.add_retrieval_reg_head:
            errors.append("paper_core_mode is incompatible with model.add_retrieval_reg_head.")
        if config.model.ct_mil:
            errors.append("paper_core_mode is incompatible with model.ct_mil.")
        if config.model.using_attnmil:
            errors.append("paper_core_mode expects its own dual-stream pooling and should not enable using_attnmil.")

        forbidden_loss_names = (
            "lambda_cls",
            "lambda_RPS",
            "lambda_categorical_ordinal_focal_loss",
            "lambda_regress",
            "lambda_consis",
            "lambda_clip",
            "lambda_bmc",
            "lambda_ranking",
        )
        for loss_name in forbidden_loss_names:
            if getattr(config.loss, loss_name, 0.0) > 0:
                errors.append(f"paper_core_mode is incompatible with loss.{loss_name} > 0.")
    else:
        for loss_name in ("lambda_align", "lambda_paper_consistency"):
            if getattr(config.loss, loss_name, 0.0) > 0:
                errors.append(f"loss.{loss_name} requires model.paper_core_mode=True.")
        if getattr(config.model, "use_uar", False):
            errors.append("model.use_uar requires model.paper_core_mode=True.")

    if errors:
        raise ValueError("Invalid training configuration:\n- " + "\n- ".join(errors))


def build_text_inputs(batch_data, tokenizer, device, config):
    def to_device(x):
        if isinstance(x, dict):
            return {k: v.to(device, non_blocking=True) for k, v in x.items()}
        return x.to(device, non_blocking=True)

    texts = {
        'text_basic': to_device(tokenizer(batch_data["text_basic_raw"])),
        'text_ct': to_device(tokenizer(batch_data["text_ct_raw"])),
        'text_all': to_device(tokenizer(batch_data["text_all_raw"])),
    }
    if uses_paper_core_mode(config):
        fine_grained_texts, fine_grained_mask = build_fine_grained_text_groups(
            batch_data["text_ct_raw"],
            max_sentences=getattr(config.model, "max_fine_grained_sentences", 8),
        )
        texts['text_fine_grained'] = tokenize_grouped_texts(fine_grained_texts, tokenizer, device)
        texts['text_fine_grained_mask'] = fine_grained_mask.to(device, non_blocking=True)
    else:
        texts['text_fine_grained'] = None
        texts['text_fine_grained_mask'] = None
    if needs_categorize_text(config):
        texts['text_categorize'] = to_device(tokenizer(generate_categorize_text()))
    else:
        texts['text_categorize'] = None
    return texts


def split_fine_grained_report(report_text):
    if report_text is None:
        return []
    parts = re.split(r"[;；。.\n]+", report_text)
    return [part.strip() for part in parts if part.strip()]


def build_fine_grained_text_groups(batch_texts, max_sentences):
    grouped_texts = []
    mask = []
    for report_text in batch_texts:
        sentences = split_fine_grained_report(report_text)
        if not sentences:
            fallback = report_text.strip() if isinstance(report_text, str) and report_text.strip() else "Abnormal brain CT scan"
            sentences = [fallback]
        sentences = sentences[:max_sentences]
        valid_num = len(sentences)
        grouped_texts.append(sentences + [""] * (max_sentences - valid_num))
        mask.append([True] * valid_num + [False] * (max_sentences - valid_num))
    return grouped_texts, torch.tensor(mask, dtype=torch.bool)


def tokenize_grouped_texts(grouped_texts, tokenizer, device):
    flat_texts = [text for sample_texts in grouped_texts for text in sample_texts]
    tokenized = tokenizer(flat_texts)
    batch_size, group_size = len(grouped_texts), len(grouped_texts[0])
    if isinstance(tokenized, dict):
        return {
            key: value.to(device, non_blocking=True).reshape(batch_size, group_size, *value.shape[1:])
            for key, value in tokenized.items()
        }
    return tokenized.to(device, non_blocking=True).reshape(batch_size, group_size, *tokenized.shape[1:])


def compute_paper_alignment_loss(fine_visual_features, fine_text_features, fine_mask, temperature):
    if fine_visual_features is None or fine_text_features is None or fine_mask is None:
        return None

    fine_visual_features = F.normalize(fine_visual_features, dim=-1)
    fine_text_features = F.normalize(fine_text_features, dim=-1)

    losses = []
    for batch_idx in range(fine_visual_features.shape[0]):
        valid_idx = fine_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        visual = fine_visual_features[batch_idx, valid_idx]
        text = fine_text_features[batch_idx, valid_idx]
        logits = torch.matmul(visual, text.transpose(0, 1)) / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        losses.append(F.cross_entropy(logits, targets))

    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_paper_consistency_loss(logic_features, global_tca_features):
    logic_features = F.normalize(logic_features, dim=-1)
    global_tca_features = F.normalize(global_tca_features, dim=-1)
    return 1 - (logic_features * global_tca_features).sum(dim=-1).mean()


def get_uar_class_names(config, expected_num_classes, dataset_class_names=None):
    class_names = list(getattr(config.model, "uar_class_names", []))
    if not class_names and dataset_class_names is not None:
        class_names = list(dataset_class_names)
    if not class_names:
        return None
    if len(class_names) != expected_num_classes:
        logging.warning(
            "Skip UAR: expected %d class names but got %d.",
            expected_num_classes,
            len(class_names),
        )
        return None
    return class_names


def encode_uar_prompt_embeddings(base_model, tokenizer, device, config, num_classes, dataset_class_names=None):
    class_names = get_uar_class_names(
        config,
        num_classes,
        dataset_class_names=dataset_class_names,
    )
    if class_names is None:
        return None

    prompt_template = getattr(config.model, "uar_prompt_template", "This CT study shows: {}.")
    prompts = [prompt_template.format(class_name) for class_name in class_names]
    prompt_tokens = tokenizer(prompts)
    if isinstance(prompt_tokens, dict):
        prompt_tokens = {
            key: value.to(device, non_blocking=True)
            for key, value in prompt_tokens.items()
        }
    else:
        prompt_tokens = prompt_tokens.to(device, non_blocking=True)
    return base_model.encode_projected_text(prompt_tokens)


def refine_multilabel_probs_with_uar(base_model, raw_probs, slice_features, class_text_embeds, config):
    semantic_scores = base_model.compute_prompt_semantic_scores(
        slice_features,
        class_text_embeds,
        temperature=getattr(config.model, "uar_temperature", 0.07),
    )
    uncertainty = 1 - torch.abs(2 * raw_probs - 1).pow(getattr(config.model, "uar_alpha", 2.0))
    mixing_weight = torch.clamp(
        getattr(config.model, "uar_lambda", 0.5) * uncertainty,
        min=0.0,
        max=1.0,
    )
    return (1 - mixing_weight) * raw_probs + mixing_weight * semantic_scores


# ------------------------- DDP utils -------------------------
# def setup_distributed():
#     """
#     初始化分布式环境（torchrun 提供的环境变量）
#     """
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#         local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     else:
#         # 单卡/非分布式回退
#         rank, world_size, local_rank = 0, 1, 0

#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(backend="nccl", init_method="env://", timeout=torch.distributed.timedelta(seconds=1800))
#     return rank, world_size, local_rank

def setup_distributed():
    import os
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=1800)  # 或者去掉该参数
        )
    else:
        # 非分布式回退
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def filter_model_kwargs(model_cfg, model_cls):
    valid_keys = set(inspect.signature(model_cls.__init__).parameters.keys()) - {"self", "clip_backbone", "out_cls"}
    return {k: v for k, v in model_cfg.items() if k in valid_keys}


def set_seed(seed: int, rank: int = 0):
    if seed is None:
        return
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- Train / Val step -------------------------
def train_one_epoch(
    model, train_loader, optimizer, scheduler, criterion_pack, device, epoch, config, writer, rank, scaler
):
    model.train()
    loss_average = AvgMeter()
    step_in_epoch = 0
    pred_icp_train, gt_icp_train = [], []

    # 打印学习率（仅主进程）
    if is_main_process():
        # if config.model.freeze_backbone:
        #     logging.info(f"lr: {optimizer.param_groups[0]['lr']}")
        #     print(f"lr: {optimizer.param_groups[0]['lr']}")
        # else:
        #     logging.info(
        #         f"lr_backbone: {optimizer.param_groups[0]['lr']}, lr_others: {optimizer.param_groups[1]['lr']}"
        #     )
        #     print(
        #         f"lr_backbone: {optimizer.param_groups[0]['lr']}, lr_others: {optimizer.param_groups[1]['lr']}"
        #     )

        log_lrs(optimizer, prefix=f"Epoch {epoch}")
        if criterion_pack.get("bmc", None) is not None:
            print(f"BMC_noise_sigma: {optimizer.param_groups[-1]['params']}, lr: {optimizer.param_groups[-1]['lr']}")

    total_idx = len(train_loader.dataset)

    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        data = resize(batch_data["image"]).float().to(device, non_blocking=True)
        label = batch_data["label"].to(device, non_blocking=True) if uses_cls_targets(config) else None
        target_onehot = batch_data['label_onehot'].to(device, non_blocking=True) if uses_onehot_targets(config) else None
        icp_value = None
        if needs_scalar_targets(config):
            icp_value = batch_data['icp_value_norm'].float().view(-1, 1).to(device, non_blocking=True)

        # 准备文本
        tokenizer = criterion_pack["tokenizer"]
        texts = build_text_inputs(batch_data, tokenizer, device, config)

        logits, video_features_pooled, attn_weights, text_embeds, text_embeds_categorize, img_embeds = model(texts, data)

        # ====== 计算总损失 ======
        loss_dict = {}
        total_loss = 0.0
        paper_aux = logits.get('paper_aux') if uses_paper_core_mode(config) else None

        if config.loss.lambda_cls > 0 and config.model.add_cls_head:
            loss_cls = criterion_pack["ce"](logits['cls'], label) * config.loss.lambda_cls
            loss_dict['loss_cls'] = loss_cls.item()
            total_loss += loss_cls

        if config.loss.lambda_RPS > 0 and config.model.add_cls_head:
            loss_rps = criterion_pack["rps"](logits['cls'], label) * config.loss.lambda_RPS
            loss_dict['loss_RPS'] = loss_rps.item()
            total_loss += loss_rps

        if config.loss.lambda_categorical_ordinal_focal_loss > 0 and config.model.add_cls_head:
            loss_cof = criterion_pack["cof"](logits['cls'], target_onehot) * config.loss.lambda_categorical_ordinal_focal_loss
            loss_dict['loss_categorical_ordinal_focal'] = loss_cof.item()
            total_loss += loss_cof

        if config.loss.lambda_regress > 0 and config.model.add_reg_head:
            if config.loss.using_LDS:
                weights = batch_data['lds_weights'].to(icp_value.device, non_blocking=True)
                loss_reg = weighted_mse_loss(logits['reg'], scaler.scale(icp_value), weights=weights.unsqueeze(1)) * config.loss.lambda_regress
            else:
                loss_reg = criterion_pack["reg"](logits['reg'], scaler.scale(icp_value)) * config.loss.lambda_regress
            loss_dict['loss_reg'] = loss_reg.item()
            total_loss += loss_reg

        if config.loss.lambda_consis > 0:
            loss_consis = criterion_pack["contrastive"](img_embeds, video_features_pooled, attn_weights) * config.loss.lambda_consis
            loss_dict['loss_consis'] = loss_consis.item()
            total_loss += loss_consis

        if uses_paper_core_mode(config) and config.loss.lambda_align > 0:
            loss_align = compute_paper_alignment_loss(
                paper_aux['fine_visual_features'],
                paper_aux['fine_text_features'],
                paper_aux['fine_grained_mask'],
                config.loss.alignment_temperature,
            )
            if loss_align is not None:
                loss_align = loss_align * config.loss.lambda_align
                loss_dict['loss_align'] = loss_align.item()
                total_loss += loss_align

        if uses_paper_core_mode(config) and config.loss.lambda_paper_consistency > 0:
            loss_paper_consistency = compute_paper_consistency_loss(
                paper_aux['logic_features'],
                paper_aux['global_tca_features'],
            ) * config.loss.lambda_paper_consistency
            loss_dict['loss_paper_consistency'] = loss_paper_consistency.item()
            total_loss += loss_paper_consistency

        if config.loss.lambda_clip > 0:
            txt_all = gather_features_with_grad(text_embeds)
            img_all = gather_features_with_grad(video_features_pooled)

            # logit_scale = (model.module.logit_scale if isinstance(model, DDP) else model.logit_scale).exp()
            # B = img.size(0); rank = dist.get_rank() if dist.is_initialized() else 0
            # targets = torch.arange(B, device=img.device) + B * rank

            # logits_i2t = logit_scale * (img @ txt_all.t())
            # logits_t2i = logit_scale * (txt @ img_all.t())

            # # 用你现有的 CLIP_dis_loss 适配，或直接用 CE
            # loss_i2t = torch.nn.functional.cross_entropy(logits_i2t, targets)
            # loss_t2i = torch.nn.functional.cross_entropy(logits_t2i, targets)
            # loss_clip = 0.5 * (loss_i2t + loss_t2i) * config.loss.lambda_clip
            
            sims_batch = sim_matrix(txt_all, img_all, pooling_type='xpooling' if config.model.using_xpooling else 'avg')
            
            loss_clip = criterion_pack["clip_dis"](
                sims_batch,
                icp_value,
                model.module.logit_scale if isinstance(model, DDP) else model.logit_scale
            ) * config.loss.lambda_clip

            loss_dict['loss_clip'] = loss_clip.item()
            total_loss += loss_clip

        if config.loss.lambda_bmc > 0 and config.model.add_reg_head and (criterion_pack.get("bmc", None) is not None):
            loss_bmc = criterion_pack["bmc"](logits['reg'], scaler.scale(icp_value)) * config.loss.lambda_bmc
            loss_dict['loss_bmc'] = loss_bmc.item()
            total_loss += loss_bmc

        if config.model.add_retrieval_reg_head:
            weights = dict(ce_loss=1.0, kl_loss=1.0, reg_loss=1.0)
            losses = criterion_pack["retrieval_reg"](logits['sims_retrieval'], logits['reg'], scaler.scale(icp_value), label)
            total_loss += sum([weights[k] * loss for k, loss in losses.items()])
            loss_dict.update({k: v.item() for k, v in losses.items()})

        if config.loss.lambda_regress > 0 and config.model.ct_mil:
            loss_reg_ct_mil = criterion_pack["reg"](logits['reg_ct_mil'], scaler.scale(icp_value)) * config.loss.lambda_regress
            loss_dict['loss_reg_ct_mil'] = loss_reg_ct_mil.item()
            total_loss += loss_reg_ct_mil

            pred_icp_train.append((scaler.unscale(logits['reg_ct_mil']).squeeze().detach().cpu().numpy() + config.loaders.offset) /
                                  config.loaders.scaler * (config.loaders.max_value - config.loaders.min_value) + config.loaders.min_value)
            gt_icp_train.append((icp_value.squeeze().detach().cpu().numpy() + config.loaders.offset) /
                                config.loaders.scaler * (config.loaders.max_value - config.loaders.min_value) + config.loaders.min_value)


        if config.loss.lambda_ranking and config.model.add_reg_head:
            loss_rank = spearman_rank_loss(
                img_embeddings=img_embeds,
                icp_value=icp_value,
                tau=0.05,
                exclude_self=True,
                use_ddp_gather=True,   # DDP 建议开：让所有 GPU 的样本一起参与排名
            ) 
            loss_dict['loss_spearank'] = loss_rank.item()
            total_loss += loss_rank

        if config.model.add_multi_cls and config.loss.lambda_multi_cls >0:
            label_multi = batch_data['label_multi'].to(device, non_blocking=True).float()
            loss_multi_cls = criterion_pack['multi_cls'](logits['multi_cls'], label_multi) * config.loss.lambda_multi_cls
            loss_dict['loss_multi_cls'] = loss_multi_cls.item()
            total_loss += loss_multi_cls


        if config.model.add_reg_head or config.model.add_retrieval_reg_head:
            pred_icp_train.append(scaler.unscale(logits['reg']).squeeze().detach().cpu().numpy())
            gt_icp_train.append(icp_value.squeeze().detach().cpu().numpy())


        total_loss.backward()
        optimizer.step()


        # 日志（仅主进程）
        if is_main_process():
            if epoch == 2:
                loss_average.reset()
            loss_average.update(total_loss.data, config.loaders.batch_size)

            if batch_idx % config.trainer.print_freq == 0:
                # print(loss_dict)
                total_loss_val = log_losses(
                    loss_dict, epoch, batch_idx * config.loaders.batch_size, writer, loss_average, total_idx
                )
                step_in_epoch += 1
    if scheduler is not None:
        scheduler.step()
    # 仅主进程保存训练散点图
    if is_main_process() and tracks_scalar_predictions(config):
        plot_save_path = os.path.join(config.checkpoints_dir, config.name, f'plots/train/epoch_{epoch}.png')
        plot_and_save_scatter(gt_icp_train, pred_icp_train, epoch=epoch, save_path=plot_save_path)


@torch.no_grad()
def evaluate(model, val_loader, criterion_pack, device, epoch, config, writer, scaler):
    model.eval()
    all_clip_loss, pred, pred_score, gt = [], [], [], []
    pred_icp, gt_icp = [], []
    abn_preds, abn_gts = [], []

    tokenizer = criterion_pack["tokenizer"]
    base_model = get_base_model(model)
    uar_prompt_embeds = None
    if uses_paper_core_mode(config) and config.model.add_multi_cls and getattr(config.model, "use_uar", False):
        num_classes = base_model.multi_cls_head[-1].out_features
        dataset_class_names = getattr(val_loader.dataset, "label_names", None)
        uar_prompt_embeds = encode_uar_prompt_embeddings(
            base_model,
            tokenizer,
            device,
            config,
            num_classes,
            dataset_class_names=dataset_class_names,
        )
        if uar_prompt_embeds is None and is_main_process():
            print("UAR is enabled but no valid class-name source was found. Falling back to raw MIL probabilities.")
            logging.warning("UAR is enabled but no valid class-name source was found. Falling back to raw MIL probabilities.")

    for val_batch_id, batch_data in enumerate(val_loader):
        val_data = resize(batch_data["image"]).to(device, non_blocking=True)
        val_label = batch_data["label"].to(device, non_blocking=True) if uses_cls_targets(config) else None
        icp_value = batch_data['icp_value_norm'] if needs_scalar_targets(config) else None
        texts = build_text_inputs(batch_data, tokenizer, device, config)

        logits, video_features_pooled, attn_weights, text_embeds, text_embeds_categorize, img_embeds = model(texts, val_data)

        if config.loss.lambda_clip > 0:
            sims_batch = sim_matrix(text_embeds, video_features_pooled, pooling_type='xpooling' if config.model.using_xpooling else 'avg')
            loss_clip = CLIPLoss(sims_batch, model.module.logit_scale if isinstance(model, DDP) else model.logit_scale)
            all_clip_loss.append(loss_clip.item())

        if config.model.add_reg_head or config.model.add_retrieval_reg_head:
            pred_icp_batch = scaler.unscale(logits['reg']).squeeze().detach().cpu().numpy()
            gt_icp_batch = icp_value.squeeze().numpy() 
            pred_icp.append(pred_icp_batch)
            gt_icp.append(gt_icp_batch)

        if config.model.ct_mil:
            pred_icp_batch = scaler.unscale(logits['reg_ct_mil']).squeeze().detach().cpu().numpy()
           
            gt_icp_batch = icp_value.squeeze().numpy() 
            pred_icp.append(pred_icp_batch)
            gt_icp.append(gt_icp_batch)

        if config.model.add_cls_head:
            pred_batch, pred_score_batch, gt_batch = evaluate_predictions(logits['cls'], val_label)
            pred.append(pred_batch); pred_score.append(pred_score_batch); gt.append(gt_batch)

        if config.model.add_retrieval_reg_head:
            pred_batch, pred_score_batch, gt_batch = evaluate_predictions(logits['sims_retrieval'], val_label)
            pred.append(pred_batch); pred_score.append(pred_score_batch); gt.append(gt_batch)

        if config.model.add_multi_cls:
            pred_score_batch = torch.sigmoid(logits['multi_cls'])
            if (
                uses_paper_core_mode(config)
                and getattr(config.model, "use_uar", False)
                and uar_prompt_embeds is not None
            ):
                paper_aux = logits.get('paper_aux')
                if paper_aux is not None:
                    pred_score_batch = refine_multilabel_probs_with_uar(
                        base_model,
                        pred_score_batch,
                        paper_aux['slice_features'],
                        uar_prompt_embeds,
                        config,
                    )

            pred_score_batch = pred_score_batch.cpu().numpy()
            gt_batch = batch_data['label_multi'].cpu().numpy() #[B,C]

            abn_preds.append(pred_score_batch)
            abn_gts.append(gt_batch)

    if is_main_process():
        metrics_dict = log_metrics(
            gt, pred, pred_score, epoch, writer, all_clip_loss, pred_icp, gt_icp, abn_preds, abn_gts,
            save_dir=os.path.join(config.checkpoints_dir, config.name)
        )
    else:
        metrics_dict = None
    # 仅主进程画验证散点
    if is_main_process() and tracks_scalar_predictions(config):
        plot_save_path = os.path.join(config.checkpoints_dir, config.name, f'plots/val/epoch_{epoch}.png')
        plot_and_save_scatter(gt_icp, pred_icp, epoch=epoch, save_path=plot_save_path)

    return metrics_dict


def main():
    # 1) 解析配置
    config = default_argument_parser()
    # 2) 初始化分布式
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.multiprocessing.set_start_method("fork", force=True)
    ddp_enabled = dist.is_available() and dist.is_initialized()
    if ddp_enabled and dist.get_rank() != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass  # 禁用print

    save_path = os.path.join(config.checkpoints_dir, config.name)
    if os.path.exists(save_path) and config.name != 'debug' and is_main_process():
        timestamp = datetime.now().strftime('%Y%m%d_%H')
        config.defrost(); config.name += f"_{timestamp}"; config.freeze()
    save_path = os.path.join(config.checkpoints_dir, config.name)
    if is_main_process():
        os.makedirs(save_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(save_path, 'train.log'),
            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
            level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %I:%M:%S %p'
        )
        writer = SummaryWriter(os.path.join(save_path, 'summary'))
        logging.info(config)
    else:
        writer = None

    # 4) 随机种子（带 rank 偏移）
    manual_seed = config.get('manual_seed', None)
    set_seed(manual_seed, rank)

    # 5) 检查模型配置
    try:
        check_config(config.model)
        validate_runtime_config(config)
    except ValueError as e:
        if is_main_process():
            print(e)
        cleanup_distributed()
        return

    # 6) 加载 backbone
    clip_model, tokenizer, context_length = build_backbone(config.model, device)

    # 7) 构建模型（先放本地 rank 设备）
    model_kwargs = filter_model_kwargs(config.model, CLIPTransformer)
    model = CLIPTransformer(clip_backbone=clip_model, out_cls=len(config.loaders.cls_thresholds)+1, **model_kwargs)
    model.to(device)

    # 8) DDP 包装（注意：先 to(device) 再 DDP）
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 9) Loader（分布式采样器）
    shuffle_val = True if config.loss.lambda_clip > 0 else False
    train_loader, val_loader, train_ds, val_ds = get_loader(
        config.loaders, shuffle_val=shuffle_val, using_LDS=config.loss.using_LDS, debug=config.debug, log_dir=save_path,
        distributed=ddp_enabled, rank=rank, world_size=world_size
    )
    # scaler initialize
    if needs_scalar_targets(config):
        y_train = train_ds.targets
        scaler = YScaler(mode="logz", clip_percentiles=(config.loaders.min_value, config.loaders.max_value)).fit(y_train)
        scaler.save(os.path.join(config.checkpoints_dir, config.name, 'scaler.json'))
    else:
        scaler = None

    # 如果你的 get_loader 已经内部用了 DistributedSampler，这里什么也不做。
    # 否则（常见情况），我们替换/包装 sampler：
    # if not isinstance(train_loader.sampler, DistributedSampler):
    #     train_loader.sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    # if not isinstance(val_loader.sampler, DistributedSampler):
    #     # 验证通常不 shuffle
    #     val_loader.sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    if is_main_process():
        logging.info(f'Train samples(total): {len(train_ds)}, Val samples(total): {len(val_ds)}')
        print(f'Train samples(total): {len(train_ds)}, Val samples(total): {len(val_ds)}')
        print('debug mode:', config.debug)
        print(f"active_paths: {format_active_paths(config)}")

    # 10) 优化器 & 调度器 & 损失器打包
    # optimizer = torch.optim.Adam(
    #     separate_clip_params(model, config.optimizer.clip_lr, config.optimizer.other_lr,
    #                          config.model.freeze_backbone, config.model.freeze_text_backbone),
    #     lr=config.optimizer.other_lr, weight_decay=config.optimizer.weight_decay,
    #     betas=(config.optimizer.beta1, 0.999)
    # )
    param_groups = separate_clip_params(
        model,
        clip_lr=config.optimizer.clip_lr,
        other_lr=config.optimizer.other_lr,
        weight_decay=config.optimizer.weight_decay,
        freeze_all_backbone=config.model.freeze_backbone,
        freeze_text_backbone=config.model.freeze_text_backbone,
        )

    optimizer = torch.optim.AdamW(  # 若想继续用 Adam，也行：torch.optim.Adam(param_groups, lr=..., betas=...)
        param_groups,
        lr=config.optimizer.other_lr,                 # 作为默认 lr，组内会覆盖
        betas=(config.optimizer.beta1, 0.999),
        weight_decay=0.0,                             # 设为0，避免覆盖组内设置
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.scheduler.lr_decay_iters, gamma=0.1)

    criterion_pack = {
        "tokenizer": tokenizer,
        "context_length": context_length,
        "ce": nn.CrossEntropyLoss().to(device),
        # "ce": FocalLoss().to(device),
        "rps": RPS(alpha_ce=0, beta_rps=10).to(device),
        "reg": nn.MSELoss().to(device),  
        # "reg": HuberCCCLoss().to(device), # 如需 Huber/CCC，可在此替换
        "clip_dis": CLIP_dis_loss().to(device),
        "retrieval_reg": Retrieval_Regression_Loss().to(device),
        "contrastive": ContrastiveLoss(temperature=config.loss.consis_temparature).to(device),
        'multi_cls': AsymmetricLossOptimized()
    }
    if config.loss.lambda_bmc > 0:
        criterion_pack["bmc"] = BMCLoss(init_noise_sigma=10.).to(device)
        optimizer.add_param_group({'params': criterion_pack["bmc"].noise_sigma, 'lr': 0.01, 'name': 'noise_sigma'})

    # 11) 训练循环
    best_auc, best_mae = 0.0, 1e9
    best_auc_epoch, best_mae_epoch = 0, 0
    best_micro_auc, best_micro_auc_epoch = 0.0, 0

    for epoch in range(1, config.scheduler.n_epochs + 1):
        # 分布式 sampler 设置 epoch（确保各 rank 切分/打乱一致）
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, optimizer, scheduler, criterion_pack, device, epoch, config, writer, rank, scaler)

        # 同步所有进程后评价（避免有人还在跑）
        if ddp_enabled:
            dist.barrier()

        metrics_dict = evaluate(model, val_loader, criterion_pack, device, epoch, config, writer, scaler)
        model_save_path = os.path.join(config.checkpoints_dir, config.name, f'weights/epoch_{epoch}.pth')

        if is_main_process():
            metric_mode = checkpoint_metric_mode(config)
            if metric_mode == "regression":
                model_save_path = model_save_path.replace('.pth', f"_mae_{metrics_dict['mae']:.3f}_r2_{metrics_dict['r2']:.3f}.pth")

            save_flag = False
            if epoch % config.trainer.save_epoch_freq == 0:
                save_flag = True
            elif metric_mode == "regression":
                if metrics_dict['mae'] < best_mae:
                    best_mae = metrics_dict['mae']
                    best_mae_epoch = epoch
                    model_save_path = model_save_path.replace('.pth', f'_best_mae_{best_mae:.3f}.pth')
                    save_flag = True
            elif metric_mode == "classification":
                model_save_path = model_save_path.replace('.pth', f"_auc_{metrics_dict['auc']:.3f}.pth")
                if metrics_dict['auc'] > best_auc:
                    best_auc = metrics_dict['auc']
                    best_auc_epoch = epoch
                    model_save_path = model_save_path.replace('.pth', f"_best_auc_{best_auc:.3f}.pth")
                    save_flag = True
            elif metric_mode == "multilabel":
                model_save_path = model_save_path.replace('.pth', f"_micro_auc_{metrics_dict['micro_AUC']:.3f}.pth")
                if metrics_dict['micro_AUC'] > best_micro_auc:
                    best_micro_auc = metrics_dict['micro_AUC']
                    best_micro_auc_epoch = epoch
                    model_save_path = model_save_path.replace('.pth', f"_best_micro_auc_{best_micro_auc:.3f}.pth")
                    save_flag = True
            elif metric_mode == "clip":
                model_save_path = model_save_path.replace('.pth', f"_cliploss_{metrics_dict['clip_loss_eval']:.3f}.pth")
                save_flag = True  # 若你想按 clip loss 也周期保存

            if config.save_ckpt and save_flag:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                state_dict = model.module.state_dict() if ddp_enabled else model.state_dict()
                torch.save(state_dict, model_save_path)
                print('save ckpt of epoch', epoch)

        # 等待 rank0 存完
        if ddp_enabled:
            dist.barrier()

    # 结束
    if is_main_process():
        print(
            f"Training finished. Best MAE {best_mae:.3f} @ epoch {best_mae_epoch}, "
            f"Best AUC {best_auc:.3f} @ epoch {best_auc_epoch}, "
            f"Best Micro-AUC {best_micro_auc:.3f} @ epoch {best_micro_auc_epoch}"
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
