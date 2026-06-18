import logging
import sys
import importlib
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, classification_report
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import os
import scipy.stats as stats
import cv2
from sklearn import metrics 
from sklearn.metrics import hamming_loss, roc_auc_score, average_precision_score, multilabel_confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
import scipy.stats as stats
import csv

loggers = {}


def append_metrics_csv(metrics_dict, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics_epoch.csv")
    row = {"epoch": epoch}

    for key, value in metrics_dict.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            row[key] = float(value)

    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
def flatten(x):
    """
    扁平化收集成一维 numpy.ndarray[float]：
    - 支持 Python/NumPy 标量、0/多维 ndarray、Torch 张量（含 0 维）、嵌套 list/tuple
    - None 会被忽略
    """
    out = []

    def _push(v):
        if v is None:
            return
        # Python/NumPy 标量
        if isinstance(v, (int, float, np.number)):
            out.append(float(v))
        # Torch 张量
        elif isinstance(v, torch.Tensor):
            if v.numel() == 1:
                out.append(float(v.item()))
            else:
                out.extend(v.detach().cpu().reshape(-1).tolist())
        # NumPy 数组
        elif isinstance(v, np.ndarray):
            if v.ndim == 0:          # 0维：np.array(6)
                out.append(float(v.item()))
            else:                    # 多维：摊平成一维
                out.extend(v.reshape(-1).tolist())
        # 嵌套序列
        elif isinstance(v, (list, tuple)):
            for u in v:
                _push(u)
        else:
            # 尝试转成 ndarray 兜底
            try:
                arr = np.asarray(v)
                if arr.ndim == 0:
                    out.append(float(arr.item()))
                else:
                    out.extend(arr.reshape(-1).tolist())
            except Exception:
                raise TypeError(f"Unsupported data type: {type(v)}")

    _push(x)
    return np.asarray(out, dtype=float)

# def flatten(list):
#     print(list)
#     flattened_list = []
#     for data in list:
#         if isinstance(data, (float, int, np.float64)):  # 标量（Python 原生或 NumPy 标量）
#             flattened_list.append(float(data))
#         elif isinstance(data, torch.Tensor):  # PyTorch 张量
#             if data.ndim == 0:  # 标量张量
#                 flattened_list.append(data.item())
#             else:  # 向量或数组
#                 flattened_list.extend(data.numpy().tolist())
#         elif isinstance(data, np.ndarray):  # NumPy 数组
#             flattened_list.extend(data.tolist())
#         elif isinstance(data, np.float32):  # NumPy float32
#             flattened_list.append(float(data))
#         else:
#             raise TypeError(f"Unsupported data type: {type(data)}")
#     return np.array(flattened_list)

def get_logger(name, level=logging.INFO, file=None):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        
        # stream_handler = logging.StreamHandler(sys.stdout)
        handler2 = logging.FileHandler(filename=file)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        # stream_handler.setFormatter(formatter)
        handler2.setFormatter(formatter)
        # logger.addHandler(stream_handler)
        logger.addHandler(handler2)

        loggers[name] = logger

        return logger


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')

def save_training_data(imgs,save_path):
    '''
    imgs: [T,H,W]
    '''
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img = imgs[imgs.shape[0]//2]
    img = (img + 1)/2
    cv2.imwrite(save_path,np.uint8(img*255))


def plot_and_save_scatter(gt_score, pred_score, epoch, save_path):
    gt_score = flatten(gt_score)
    pred_score = flatten(pred_score)
    # gt_score, pred_score = np.array(gt_score), np.array(pred_score)
    Num = len(gt_score)
    # Calculate MAE and R2
    mae = mean_absolute_error(gt_score, pred_score)
    r2 = r2_score(gt_score, pred_score)
    corr,p = stats.pearsonr(pred_score, gt_score)
    max_icp = max([gt_score.max(), pred_score.max()])
    min_icp = min([gt_score.min(), pred_score.min()])
    # Set the seaborn style
    sns.set(style="whitegrid")

    # Create a figure and a 1x2 grid layout for the two plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot (Left plot)
    axs[0].scatter(gt_score, pred_score, color='#6fa3f7', alpha=0.7, label='Predicted vs Ground Truth', edgecolor='black')
    axs[0].plot([int(min_icp)-5, int(max_icp)+5], [int(min_icp)-5, int(max_icp)+5], color='#2b2b2b', linestyle='-', linewidth=2, label='Ideal Line (No Error)')
    axs[0].text(int(max_icp)//2, 5, f"MAE = {mae:.2f}; $R^2$ = {r2:.2f}", color='#4c4c4c', fontsize=15)
    axs[0].text(int(max_icp)//2, 2, f"corr = {corr:.2f}; p = {p:.2f}", color='#4c4c4c', fontsize=15)
    # axs[0].text(int(max_icp)//2, 0, f"Num = {Num:.2f}", color='#4c4c4c', fontsize=15)

    axs[0].set_xlabel('Ground Truth (icp_GT)', fontsize=14, color='#333333')
    axs[0].set_ylabel('Predicted (icp_Pred)', fontsize=14, color='#333333')
    axs[0].set_title(f'Scatter Plot of icp_GT vs icp_Pred of Epoch[{epoch}]', fontsize=16, color='#2b2b2b')
    axs[0].grid(color='lightgrey', linestyle='--', linewidth=0.5)
    axs[0].legend(fontsize=12, loc='upper left')
    axs[0].set_xlim(int(min_icp)-5, int(max_icp)+5)
    axs[0].set_ylim(int(min_icp)-5, int(max_icp)+5)

    # Histogram with kde (Right plot)
    sns.histplot(gt_score, kde=True, bins='auto', color='skyblue', stat='count', linewidth=0, ax=axs[1], label=f'gt_score [{Num}]')
    sns.histplot(pred_score, kde=True, bins='auto', color='orange', stat='count', linewidth=0, ax=axs[1], label=f'pred_score [{Num}]')
    
    axs[1].set_title(f'Distribution of gt_score and pred_score of Epoch[{epoch}]', fontsize=16)
    axs[1].set_xlabel('Value', fontsize=14)
    axs[1].set_ylabel('Count', fontsize=14)
    axs[1].legend(title='Legend', fontsize=12)

    plt.tight_layout()
    # 保存图表到指定路径
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):  
        os.makedirs(save_dir)
    save_path = save_path.replace('.png',f'_{mae:.3f}_{corr:.3f}.png')
    plt.savefig(save_path)
    plt.close()
    
import torch.distributed as dist
try:
    from torch.distributed.nn.functional import all_gather as all_gather_with_grad
except Exception:
    all_gather_with_grad = None  # 低版本 fallback: 下面给兜底方案

def gather_features_with_grad(x):
    """
    x: (B, D) on each rank
    return: (B_total, D) concatenated across all ranks
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x
    if all_gather_with_grad is not None:
        y_list = all_gather_with_grad(x)   # list of (B_r, D), with grad w.r.t local x
        return torch.cat(y_list, dim=0)
    # ---- fallback（无梯度跨 rank，仅本地 x 保留梯度就够用时可用）----
    xs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x.contiguous())
    y = torch.cat(xs, dim=0)
    # 注意：fallback 对远端样本不保梯度（一般也不需要）
    return y



def sim_matrix(text_embeds, vid_embeds_pooled, pooling_type='avg'):
    """
    Computes the similarity matrix using pooled video frames
    Input
        text_embeds: num_texts x E
        vid_embeds_pooled: num_vids x num_texts x E   or  num_vids x E
    Output
        sims: num_texts x num_vids
    """

    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        # num_vids x embed_dim  
        # num_texts x embed_dim
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    elif pooling_type == 'xpooling':
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)

        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)
    
    else: 
        print('no this pooling type')

    return sims

def resize(img, size = 224):
    b, c, n, h, w = img.size()
    new_img = []
    for i in range(b):
        im = img[i, :, :, :, :]
        im = F.interpolate(im, size=[size, size], mode='bilinear', align_corners=True)
        new_img.append(im.unsqueeze(0))
    new_img = torch.cat(new_img, dim=0)
    return new_img

# def separate_clip_params(model, clip_lr=1e-6, other_lr=1e-5, freeze_all_backbone=False,freeze_text_backbone=False):
#     other_params = []
#     clip_img_params = []
#     clip_text_params = []
#     # print('trainable parameters: ')
#     for pname, p in model.named_parameters():
#         if 'clip_backbone' in pname and 'text' not in pname and 'VPT' not in pname and 'lora' not in pname:
#             clip_img_params += [p]
#             if freeze_all_backbone:
#                 p.requires_grad = False
#             # else:
#             #     print(pname,end=" ")
#         elif 'clip_backbone' in pname and 'text' in pname and 'VPT' not in pname and 'lora' not in pname:
#             clip_text_params += [p]
#             if freeze_all_backbone:
#                 p.requires_grad = False
#             else:
#                 if freeze_text_backbone:
#                     p.requires_grad = False
#                 # else:
#                 #     print(pname,end=" ")
#         else:
#             other_params += [p]
#             # print(pname)
    
#     if freeze_all_backbone:
#         return other_params
#     else:
#         if freeze_text_backbone:
#             params = [
#             {'params': clip_img_params, 'lr': clip_lr},
#             {'params': other_params},
#         ]
#         else:
#             clip_params = clip_img_params + clip_text_params
#             params = [
#                 {'params': clip_params, 'lr': clip_lr},
#                 {'params': other_params},
#             ]
#         return params

def separate_clip_params(
    model,
    clip_lr=1e-6,
    other_lr=1e-5,
    weight_decay=1e-3,
    freeze_all_backbone=False,
    freeze_text_backbone=False,
):
    # 按 (模块类别 × decay/no_decay) 六类收集
    clip_img_decay,  clip_img_no_decay  = [], []
    clip_txt_decay,  clip_txt_no_decay  = [], []
    other_decay,     other_no_decay     = [], []

    def is_no_decay(name, p):
        # 规则：bias、所有 Norm 层参数（LayerNorm/BatchNorm/GroupNorm等）以及1D参数不做WD
        lname = name.lower()
        return (
            p.ndim == 1
            or lname.endswith(".bias")
            or "norm" in lname
            or "bn" in lname
            or "layernorm" in lname
            or "batchnorm" in lname
        )

    for pname, p in model.named_parameters():
        # 先按冻结规则处理
        if 'clip_backbone' in pname:
            # 区分 text 与 image
            is_text = ('text' in pname)
            # 需要冻结？
            if freeze_all_backbone or (is_text and freeze_text_backbone):
                p.requires_grad = False

        if not p.requires_grad:
            continue  # 跳过冻结参数

        # 按模块类别分拣（排除 VPT / lora 可在此处加条件）
        if ('clip_backbone' in pname) and ('VPT' not in pname) and ('lora' not in pname):
            if 'text' in pname:
                (clip_txt_no_decay if is_no_decay(pname, p) else clip_txt_decay).append(p)
            else:
                (clip_img_no_decay if is_no_decay(pname, p) else clip_img_decay).append(p)
        else:
            (other_no_decay if is_no_decay(pname, p) else other_decay).append(p)

    # 组装 param_groups（空组不加入）
    groups = []

    def add_group(params, lr, wd):
        if params:  # 只在非空时加入
            groups.append({"params": params, "lr": lr, "weight_decay": wd})

    # clip image
    add_group(clip_img_decay,    clip_lr,  weight_decay)
    add_group(clip_img_no_decay, clip_lr,  0.0)
    # clip text（可能被冻结）
    add_group(clip_txt_decay,    clip_lr,  weight_decay)
    add_group(clip_txt_no_decay, clip_lr,  0.0)
    # others
    add_group(other_decay,       other_lr, weight_decay)
    add_group(other_no_decay,    other_lr, 0.0)

    return groups

# import logging

def log_lrs(optimizer, prefix: str = "", log_level=logging.INFO, show_count=True):
    """
    自动遍历 optimizer.param_groups 打印学习率等信息。
    - 支持可选 'name' 字段；没有则用 g{idx}
    - 显示 lr / weight_decay / (可选)参数个数
    - 同时 logging 和 print 输出
    """
    lines = []
    for i, pg in enumerate(optimizer.param_groups):
        name = pg.get("name", f"g{i}")
        lr   = pg.get("lr", None)
        wd   = pg.get("weight_decay", None)
        if show_count:
            try:
                nparams = sum(p.numel() for p in pg.get("params", []) if getattr(p, "requires_grad", False))
                line = f"{name}: lr={lr:.3e}, wd={wd:.3e}, params={nparams}"
            except Exception:
                line = f"{name}: lr={lr:.3e}, wd={wd:.3e}"
        else:
            line = f"{name}: lr={lr:.3e}, wd={wd:.3e}"
        lines.append(line)

    msg = (prefix + " " if prefix else "") + " | ".join(lines)
    logging.log(log_level, msg)
    print(msg)


def log_losses(loss_dict, epoch, batch_idx, writer, loss_average, total_idx):
    """
    Log and print the losses.
    """
    total_loss = sum(loss_dict.values()) 
    loss_info = ', '.join([f'{key} = {value:.3f}' for key, value in loss_dict.items()])
    
    print(f'Epoch: {epoch}, Batch: {batch_idx}/{total_idx}, {loss_info}, loss_total = {total_loss:.3f}, loss_total_average = {loss_average.show():.3f}')
    logging.info(f'Epoch: {epoch}, Batch: {batch_idx}/{total_idx}, {loss_info}, loss_total = {total_loss:.3f}, loss_total_average = {loss_average.show():.3f}')
    writer.add_scalars('Loss', {key: torch.tensor(min(10,value)) for key, value in loss_dict.items()}, global_step=batch_idx+total_idx*(epoch-1))
    writer.add_scalars('Loss_epoch', {'loss_total':loss_average.show()}, global_step=epoch)
    
    return total_loss

def log_metrics(gt=[], pred=[], pred_score=[], epoch=0, writer=None, all_clip_loss=[],pred_icp=[],gt_icp=[], abn_preds=[], abn_gts=[], save_dir=None):
    """
    计算并打印模型的评估指标accuracy, precision, recall, f1, auc。
    """
    
    metrics_dict = {}
    if len(gt)>0:
        gt, pred, pred_score = flatten(gt), flatten(pred), flatten(pred_score)
        acc = metrics.accuracy_score(gt, pred)

        if np.unique(gt).shape[0] == 2:
            auc = roc_auc_score(gt, pred_score)
            precision = metrics.precision_score(gt, pred)
            recall = metrics.recall_score(gt, pred)
            f1 = metrics.f1_score(gt, pred)
            spe = metrics.recall_score(gt, pred, pos_label=0)
        else:
            # For multi-class classification, use 'ovr' (one-vs-rest) strategy  
            auc = roc_auc_score(gt, pred_score, multi_class='ovr', average='micro')
            precision = metrics.precision_score(gt, pred, average='macro')
            recall = metrics.recall_score(gt, pred, average='macro')
            f1 = metrics.f1_score(gt, pred, average='macro')
            spe = metrics.recall_score(gt, pred, pos_label=0, average='macro')
        # auc = metrics.roc_auc_score(gt, pred_score, average='macro', multi_class='ovr')
        metrics_dict['acc'] = acc
        metrics_dict['precision'] = precision
        metrics_dict['recall'] = recall
        metrics_dict['f1'] = f1
        metrics_dict['spe'] = spe
        metrics_dict['auc'] = auc
        print(metrics.confusion_matrix(gt, pred))
        print(classification_report(gt, pred, target_names=np.unique(gt).astype(str)))
    if len(all_clip_loss) > 0 :
        metrics_dict['clip_loss_eval'] = np.mean(all_clip_loss)
    if len(gt_icp) > 0 :
        gt_icp, pred_icp = flatten(gt_icp), flatten(pred_icp)
        metrics_dict['mae'] = mean_absolute_error(gt_icp,pred_icp)
        metrics_dict['r2'] = r2_score(gt_icp,pred_icp)
    if len(abn_gts) > 0:
        gts = np.concatenate(abn_gts, axis=0)
        scores = np.concatenate(abn_preds, axis=0)
        ml_metrics = calculate_multilabel(gts, scores, threshold=0.5)
        print("[Multilabel] micro:")
        logging.info("[Multilabel] micro:")
        for k, v in ml_metrics["micro"].items():
            print(f"  {k}: {v}")
            logging.info(f"  {k}: {v}")
        print("[Multilabel] macro:")
        logging.info("[Multilabel] macro:")
        for k, v in ml_metrics["macro"].items():
            if isinstance(v, dict):
                logging.info(f"  {k}:")
                for kk, vv in v.items():
                    logging.info(f"     {kk}: {vv}")
                continue
            else:
                print(f"  {k}: {v}")
                logging.info(f"  {k}: {v}")
        metrics_dict['micro_AUC'] = ml_metrics["micro"]['auc']
        metrics_dict['macro_AUC'] = ml_metrics["macro"]['AUC']
        metrics_dict['micro_hamming_loss'] = ml_metrics["micro"]['hamming_loss']
        metrics_dict['micro_sensitivity'] = ml_metrics["micro"]['sensitivity']
        metrics_dict['micro_specificity'] = ml_metrics["micro"]['specificity']
        metrics_dict['macro_AP'] = ml_metrics["macro"]['AP']

    metrics_info = ', '.join([f'{key} = {value:.3f}' for key, value in metrics_dict.items()])
    print(f'Epoch: {epoch} - Validation: {metrics_info}')
    logging.info(f'Epoch: {epoch} - Validation: {metrics_info}')
    writer.add_scalars('Metrics', {key: torch.tensor(value) for key, value in metrics_dict.items()}, global_step=epoch)
    if save_dir is not None:
        append_metrics_csv(metrics_dict, epoch, save_dir)
    
    return metrics_dict

def evaluate_predictions(logits, val_target):

    out = torch.softmax(logits, dim=1)
    predicted = out.argmax(dim=1)
    
    pred = predicted.cpu().numpy()  
    pred_score = out.cpu().numpy() if out.shape[1] > 2 else out[:,1].cpu().numpy()
    gt = val_target.cpu().numpy()  

    return pred, pred_score, gt

def check_config(config):
    """
    检查 PEFT / finetune 配置是否互斥。
    如果两者或以上为True，则抛出一个错误。
    """
    true_count = sum([
        (not config.freeze_backbone),
        config.use_lora,
        config.use_vpt,
        getattr(config, "use_clip_adapter", False),
    ])

    if true_count > 1:
        raise ValueError(
            "Error: only one of full finetuning, 'use_lora', 'use_vpt', or "
            "'use_clip_adapter' can be enabled at the same time."
        )
    else:
        print("Configuration is valid.")

# def separate_clip_params(model, clip_lr=1e-6, other_lr=1e-5, freeze_all_backbone=False,freeze_text_backbone=False):
#     clip_params = []
#     other_params = []
#     print('trainable parameters: ')
#     for pname, p in model.named_parameters():
#         if 'clip_backbone' in pname and 'VPT' not in pname and 'lora' not in pname:
#             clip_params += [p]
#             if freeze_all_backbone:
#                 p.requires_grad = False
#             else:
#                 print(pname)
#         else:
#             other_params += [p]
#             print(pname)
    
#     if freeze_all_backbone:
#         return other_params
#     else:
#         params = [
#             {'params': clip_params, 'lr': clip_lr},
#             {'params': other_params},
#         ]
#         return params

def calculate_recall_at_k_for_diag(similarity_matrix, k=1):
    """
    sims: num_texts x num_vids
    计算 Recall@K 指标（对角线为正样本）
    :param similarity_matrix: 相似度矩阵 (query_size, gallery_size)
    :param k: 取前 k 个结果
    :return: text2img - Recall@K 
    """
    query_size = similarity_matrix.shape[0] # query is text
    correct_at_k = 0

    for i in range(query_size):
        top_k_indices = np.argsort(similarity_matrix[i])[::-1][:k]  # 取相似度最高的 k 个索引
        if i in top_k_indices:  # 检查对角线的正样本是否在前 k 个结果中
            correct_at_k += 1

    return correct_at_k / query_size


def calculate_mean_average_precision_for_diag(similarity_matrix):
    """
    计算 Mean Average Precision (mAP)（对角线为正样本）
    :param similarity_matrix: 相似度矩阵 (query_size, gallery_size)
    :return: mAP 值
    """
    query_size = similarity_matrix.shape[0]
    average_precisions = []

    for i in range(query_size):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]  # 按相似度降序排序
        relevant_rank = np.where(sorted_indices == i)[0][0] + 1  # 正样本的排名（从 1 开始）
        average_precisions.append(1 / relevant_rank)

    return np.mean(average_precisions)

def generate_categorize_text():
    bins = [
        '7-15',
        '16-20',
        '21-40',
        '>40'
    ]
    categories = ['Normal Intracranial Pressure',
                    'Mildly Elevated Intracranial Pressure',
                    'Moderately Elevated Intracranial Pressure',
                    'Severely Elevated Intracranial Pressure']
    
    templates = []
    for i in range(len(bins)):
        templates.append('Brain CT imaging indicates {} ({} mmHg).'.format(categories[i],bins[i]))
    return templates

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))
    

def calculate_multilabel(   
    labels: np.ndarray,
    outputs_cls: np.ndarray,
    threshold: float = 0.5,
) -> dict:

    micro_metrics = evaluate_multilabel_micro_auc(labels, outputs_cls)
    macro_metrics_cls = calculate_multilabel_metrics(labels, (outputs_cls>0.5).astype(int), outputs_cls)

    return {
        **micro_metrics,
        'macro': macro_metrics_cls,
    }

    return None

def calculate_multilabel_metrics(y_true, y_pred, y_proba):
    """多标签八分类指标计算"""
    # 逐类别计算混淆矩阵
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    # 初始化存储
    macro_metrics = {'AUC': 0., 'AP': 0.}
    class_metrics = {}
    
    for i in range(y_true.shape[1]):
        # 单类别指标
        tn, fp, fn, tp = cm[i].ravel()
        # 安全计算AUC和AP
        try:
            auc = roc_auc_score(y_true[:, i], y_proba[:, i])
        except ValueError:
            auc = 0.5  # 如果无法计算AUC（如所有样本属于同一类）
        
        try:
            ap = average_precision_score(y_true[:, i], y_proba[:, i])
        except ValueError:
            ap = 0.0
        
        if np.isnan(auc):
            auc = 0.5

        # 更新宏平均
        macro_metrics['AUC'] += auc
        macro_metrics['AP'] += ap
        
        # 记录各分类结果
        class_metrics[f'cls_{i}'] = {
            f'Class_{i}_AUC': auc,
            f'Class_{i}_AP': ap,
            f'Class_{i}_TP': tp,
            f'Class_{i}_FP': fp,
            f'Class_{i}_FN': fn,
            f'Class_{i}_TN': tn,
            f'Class_{i}_Sen': tp / (tp + fn) if (tp + fn) > 0 else 0,
            f'Class_{i}_Spe': tn / (tn + fp) if (tn + fp) > 0 else 0,
            f'Class_{i}_Acc': (tp + tn) / (tp + tn + fp + fn)
        }
    
    # 计算宏平均
    num_classes = y_true.shape[1]
    macro_metrics = {k: v/num_classes for k, v in macro_metrics.items()}
    
    # 合并结果
    return {**macro_metrics, **class_metrics}
    # return macro_metrics


def evaluate_multilabel_micro_auc(
    labels: np.ndarray,
    outputs_cls: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    评估多标签分类任务的两个模型输出，计算微平均 Sensitivity、Specificity 和 AUC。

    参数:
        labels (np.ndarray): 真实标签，形状 [N, C]，元素为 0/1。
        outputs_cls (np.ndarray): 模型输出1（如分类头），形状 [N, C]，概率值。
        outputs_prot (np.ndarray): 模型输出2（如原型头），形状 [N, C]，概率值。
        threshold (float): 概率转二进制标签的阈值（仅用于 Sensitivity/Specificity）。

    返回:
        dict: 包含两个输出的评估指标（Hamming Loss, Sensitivity, Specificity, AUC）。
    """
    # 计算单个输出的评估指标
    def compute_metrics(y_true, y_prob, y_pred=None):
        # 展平为微平均计算
        y_true_flat = y_true.flatten()
        y_prob_flat = y_prob.flatten()

        # Hamming Loss（需要二值化预测）
        if y_pred is None:
            y_pred = (y_prob > threshold).astype(int)
        y_pred_flat = y_pred.flatten()
        hl = hamming_loss(y_true, y_pred)

        # 微平均 Sensitivity (Recall) 和 Specificity
        TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
        TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

        sensitivity = TP / (TP + FN + 1e-10)
        specificity = TN / (TN + FP + 1e-10)

        # AUC（无需二值化，直接基于概率）
        try:
            auc = roc_auc_score(y_true, y_prob, average="micro")
        except ValueError:
            auc = np.nan  # 处理无法计算的情况（如所有标签相同）

        return {
            "Total num": len(y_true),
            "hamming_loss": round(hl, 3),
            "sensitivity": round(sensitivity, 3),
            "specificity": round(specificity, 3),
            "auc": round(auc, 3),
        }

    # 分别评估两个输出
    preds_cls = (outputs_cls > threshold).astype(int)
    metrics_cls = compute_metrics(labels, outputs_cls, preds_cls)

    # 返回结果
    return {
        "micro": metrics_cls,
    }


import json

class YScaler:
    """
    标量回归目标缩放器
    mode="logz": 先平移到>=0，再log1p，最后z-score（推荐：右长尾分布）
    mode="robust": median/IQR 标准化（抗极端值，不做log）
    """
    def __init__(self, mode: str = "logz", eps: float = 1e-6, clip_percentiles=None):
        """
        Args:
            mode: "logz"（默认）或 "robust"
            eps: 数值稳定项
            clip_percentiles: (lo, hi) 仅在 fit() 统计时对训练集做分位裁剪（不影响实际数据）
        """
        assert mode in ("logz", "robust")
        self.mode = mode
        self.eps = float(eps)
        self.clip_percentiles = clip_percentiles
        # 统计量
        self.shift = 0.0
        self.mu = 0.0
        self.sigma = 1.0
        self.median = 0.0
        self.iqr = 1.0
        self.fitted = False

    @staticmethod
    def _to_numpy(y):
        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy().astype(np.float64).reshape(-1)
        return np.asarray(y, dtype=np.float64).reshape(-1)

    @staticmethod
    def _to_like(ref, arr):
        if isinstance(ref, torch.Tensor):
            return torch.as_tensor(arr, device=ref.device, dtype=ref.dtype).reshape(ref.shape)
        return arr.reshape(ref.shape)

    def fit(self, y_train):
        """仅用训练集拟合统计量。"""
        y = self._to_numpy(y_train)

        # 可选：为统计量计算做轻微分位裁剪（稳健对极端值）
        y_fit = y
        if self.clip_percentiles is not None:
            lo, hi = np.percentile(y, self.clip_percentiles)
            y_fit = np.clip(y_fit, lo, hi)

        if self.mode == "logz":
            # 平移：确保>=0，避免log对负数报错
            self.shift = max(0.0, -y_fit.min() + self.eps)
            y_log = np.log1p(y_fit + self.shift)
            self.mu = float(y_log.mean())
            self.sigma = float(y_log.std() + self.eps)
        else:  # robust
            q25, q75 = np.percentile(y_fit, [25, 75])
            self.median = float(np.median(y_fit))
            self.iqr = float(q75 - q25 + self.eps)

        self.fitted = True
        return self

    def scale(self, y):
        """缩放：只传 y 即可（np.ndarray 或 torch.Tensor）。"""
        assert self.fitted, "Call fit(y_train) before scale()"
        y_np = self._to_numpy(y)

        if self.mode == "logz":
            z = (np.log1p(y_np + self.shift) - self.mu) / self.sigma
        else:
            z = (y_np - self.median) / self.iqr
        # print(self._to_like(y, z).shape)
        # print('before', y)
        # print('after', z)
        return self._to_like(y, z)

    def unscale(self, y_scaled):
        """反缩放回原始尺度：只传缩放后的 y 即可。"""
        assert self.fitted, "Call fit(y_train) before unscale()"
        ys = self._to_numpy(y_scaled)

        if self.mode == "logz":
            y_log = ys * self.sigma + self.mu
            y_orig = np.expm1(y_log) - self.shift
        else:
            y_orig = ys * self.iqr + self.median
        # print('before', y_scaled)
        # print('after', y_orig)
        return self._to_like(y_scaled, y_orig)

    # 便于持久化（推理阶段复用完全相同的缩放器）
    def to_dict(self):
        return {
            "mode": self.mode,
            "eps": self.eps,
            "clip_percentiles": self.clip_percentiles,
            "shift": self.shift,
            "mu": self.mu,
            "sigma": self.sigma,
            "median": self.median,
            "iqr": self.iqr,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(mode=d["mode"], eps=d["eps"], clip_percentiles=d["clip_percentiles"])
        obj.shift = d["shift"]; obj.mu = d["mu"]; obj.sigma = d["sigma"]
        obj.median = d["median"]; obj.iqr = d["iqr"]; obj.fitted = d["fitted"]
        return obj

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)
