import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class AsymmetricLossOptimized(torch.nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum(dim=1).mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float or list): 类别平衡系数，若为float则默认对正类使用该值，负类为1-alpha
            gamma (float): 聚焦因子
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [N, 2]  —— 模型输出的原始logits（未经过softmax）
            targets: [N]   —— 0/1 二分类标签
        """
        # [N, 2] → [N, 2]
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)  # softmax 后的概率

        # 取出对应标签的概率 p_t
        targets = targets.long()
        pt = probs[torch.arange(len(probs)), targets]          # p_t
        log_pt = log_probs[torch.arange(len(probs)), targets]  # log(p_t)

        # alpha_t
        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.tensor([self.alpha, 1 - self.alpha], device=logits.device)
            alpha_t = alpha_t[targets]
        else:
            # 若传入list或tensor
            alpha_t = torch.tensor(self.alpha, device=logits.device)[targets]

        # focal loss
        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2,
                       reduction: str = "none") -> torch.Tensor:
    """
    Focal loss used in RetinaNet for dense detection.
    
    Args:
        inputs (Tensor): Predictions for each example.
        targets (Tensor): Binary classification label (0 for negative class, 1 for positive class).
        alpha (float): Weighting factor to balance positive vs. negative examples (default: 0.25).
        gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs. hard examples (default: 2).
        reduction (str): 'none' | 'mean' | 'sum' (default: 'none').
            - 'none': No reduction applied to the output.
            - 'mean': Output is averaged.
            - 'sum': Output is summed.
    
    Returns:
        Loss tensor with the specified reduction.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Invalid value for 'reduction': '{reduction}'. "
                         "Supported modes: 'none', 'mean', 'sum'")
    
    return loss

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-8

    # def forward(self, img_embeds, global_embeds, attn_weights):
    #     ''''
    #     img_embeds: [B,T,C]
    #     global_embeds: [B,C]
    #     attn_weights: [B,1,T,1]
    #     '''
    #     _, top_idx = torch.topk(attn_weights, k=10, dim=2) #[B,1,`10`,1]

    #     global_embeds = F.normalize(global_embeds, dim=1) #[B,C]
    #     img_embeds = F.normalize(img_embeds, dim=2)   #[B,T,C]

    #     contrast_features = img_embeds.view(-1, img_embeds.shape[-1])
    #     dot_product = torch.matmul(global_embeds, contrast_features.T) #[B,B*T]

    #     mask = torch.zeros_like(dot_product) #[B,B*T]
    #     for i in range(mask.shape[0]):
    #         mask[i, top_idx[i]+(i*img_embeds.shape[1])] = 1
        
    #     pos_loss = torch.exp(dot_product*mask / self.temperature).sum(dim=1) 
    #     neg_loss = torch.exp(dot_product*(1-mask) / self.temperature).sum(dim=1)
        
    #     loss = -torch.log((pos_loss + self.eps) / (neg_loss + self.eps)).mean()
    #     return loss
    def forward(self, img_embeds, global_embeds, attn_weights):
        ''''
        img_embeds: [B,T,C]
        global_embeds: [B,C]
        attn_weights: [B,1,T,1]
        '''
        b,t,c = img_embeds.shape
        top_num = min(5,t)
        _, top_idx = torch.topk(attn_weights.squeeze(), k=top_num, dim=1) #[B,top_num]
        # _, lowest_idx = torch.topk(attn_weights.squeeze(), largest=False, k=t-top_num, dim=1) #[B,t-top_num]

        # global_embeds = F.normalize(global_embeds, dim=1) #[B,C]
        # img_embeds = F.normalize(img_embeds, dim=2)   #[B,T,C]

        # contrast_features = img_embeds.view(-1, img_embeds.shape[-1]) #[B*T,C]
        # dot_product = torch.matmul(global_embeds, contrast_features.T) #[B,B*T]
        # dot_product = dot_product/self.temperature

        # top_idx += torch.arange(0, b, dtype=torch.int64)[:, None] * torch.full((1, top_num), 1, dtype=torch.int64)

        # pos_sim = dot_product.gather(1, top_idx) #[B,top_num]
        # pos_sim_ = torch.exp(pos_sim).sum(dim=(0,1)) 

        # all_sim = torch.exp(dot_product).sum(dim=(0,1)) 
            
        # loss = -torch.log((pos_sim_ + self.eps) / (all_sim + self.eps)).mean()
        
        global_embeds = F.normalize(global_embeds, dim=1).unsqueeze(1) #[B,1,C]
        img_embeds = F.normalize(img_embeds, dim=2).transpose(1,2)   #[B,C,T]

        dot_product = torch.matmul(global_embeds, img_embeds).squeeze(1) #[B,T]
        dot_product = dot_product/self.temperature

        pos_sim = dot_product.gather(1, top_idx) #[B,top_num]
        pos_sim_ = torch.exp(pos_sim).sum(dim=1) #[B]

        all_sim = torch.exp(dot_product).sum(dim=1) #[B]
            
        loss = -torch.log((pos_sim_ + self.eps) / (all_sim + self.eps)).mean()
    
        return loss

def CLIPLoss(logits, logit_scale=None):
    """
    Inputs: cosine similarities
        sims: n x n (text is dim-0)
        logit_scale: 1 x 1
    """
    # logit_scale = logit_scale.exp()
    if logit_scale is not None:
        logits = logits * logit_scale.exp()
    
    t2v_log_sm = F.log_softmax(logits, dim=1)
    t2v_neg_ce = torch.diag(t2v_log_sm)
    t2v_loss = -t2v_neg_ce.mean()

    v2t_log_sm = F.log_softmax(logits, dim=0)
    v2t_neg_ce = torch.diag(v2t_log_sm)
    v2t_loss = -v2t_neg_ce.mean()

    return (t2v_loss + v2t_loss) / 2.0

class RPS(torch.nn.Module):
    """
        alpha_ce*CE + beta_rps * rps
    Args:
        alpha_ce (float, optional): The balancing weight for CE loss. Defaults to 1.
        alpha_rps (float, optional): The balancing weight for RPS loss. Defaults to 1.
    Assumes tensors of shape:
    logits = BS x C
    target = BS, will be one-hot encoded internally using n_classes = logits.shape[1]
    """
    def __init__(self, alpha_ce: float = 1., beta_rps: float = 1., ce_weight: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.beta_rps = beta_rps
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.reduction = reduction

    def forward(self, inputs, targets):

        # loss_ce = self.cross_entropy(inputs, targets)

        num_classes = inputs.shape[1]
        labels = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes)
        probs = F.softmax(inputs, dim=1)
        rps_loss = ((torch.cumsum(labels, dim=-1) - torch.cumsum(probs, dim=-1)) ** 2).sum(dim=-1)
        if self.reduction == 'mean':
            rps_loss = rps_loss.mean()
        return self.beta_rps * rps_loss

def categorical_ordinal_focal_loss(y_pred, y_true, gamma=2., alpha=0.25, beta=0.2):
    """
    Categorical ordinal focal loss, as described in the paper: https://arxiv.org/pdf/2007.08920v1.pdf.
        
    Parameters:
      gamma -- Focusing parameter for modulating factor (1-p)
      alpha -- Weighting factor for class imbalance (similar to balanced cross entropy)
      beta -- Weighting factor for ordinal component
    """
  
    """
    :param y_true: Ground truth tensor (shape: [B, C])
    :param y_pred: Predicted probabilities from softmax (shape: [B, C])
    :return: Computed focal loss
    """
    # Normalize predictions to prevent numerical instability (ensure sum over classes is 1)
    y_pred = F.softmax(y_pred,dim=1)
    y_pred = y_pred / y_pred.sum(dim=-1, keepdim=True)
    y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)  # To prevent log(0) errors

    # Cross entropy loss (element-wise)
    cross_entropy = -y_true * torch.log(y_pred)

    # Calculate ordinal distance
    true_class = torch.argmax(y_true, dim=1)  # Argmax over class dimension to get class indices
    pred_class = torch.argmax(y_pred, dim=1)  # Argmax over class dimension for predictions
    ordinal_dist = torch.abs(true_class - pred_class)  # Absolute difference

    # Normalize ordinal distance
    weights = ordinal_dist.float() / (y_pred.size(1) - 1)

    # Focal loss modulating factor
    focal_loss = alpha * (1 - y_pred) ** gamma

    # Expand weights across all classes
    weights_expanded = weights.unsqueeze(1).expand_as(y_pred)

    # Combined loss
    combined_loss = (beta * weights_expanded + focal_loss) * cross_entropy

    # Sum over the classes (axis 1) and return the mean loss over the batch
    return combined_loss.sum(dim=1).mean()


def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(logits.device))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)


def bmc_loss_huber(pred, target, noise_scale, delta=1.0):
    """Balanced Huber Loss (BMC)
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_scale: A float number or tensor.
      delta: A float number. Hyper-parameter for Huber Loss
    Returns:
      loss: A float tensor. Balanced MAE Loss.
    """
    logits = - F.huber_loss(pred, target.T, delta=delta, reduction='none') / noise_scale   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * noise_scale.detach()  # optional: restore the loss scale, 'detach' when noise is learnable

    return loss

def bni_loss_huber(pred, target, noise_scale, bucket_centers, bucket_weights, delta=1.0):
    # Balanced Huber Loss (BNI)
    huber_term = F.huber_loss(pred, target, delta=delta, reduction='none') / noise_scale

    num_bucket = bucket_centers.shape[0]
    bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
    bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

    balancing_term = - F.huber_loss(pred.expand(-1, num_bucket), bucket_center, delta=delta, reduction='none') / noise_scale + bucket_weights.log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = huber_term + balancing_term
    loss = loss * noise_scale.detach()
    return loss.mean()



class CLIP_dis_loss(torch.nn.Module):
    def __init__(self, max_value: int = 35, min_value: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.weight_scale = max_value - min_value

    def forward(self, logits, y, logit_scale):

        '''
        logits: B*B
        y: B
        ''' 
        y_ = y.view(-1,1) #[B,1]
        weight = torch.abs(y_ - y_.t())

        logit_scale = logit_scale.exp()
        logits = logits * logit_scale
        
        t2v_weight = F.normalize(weight,dim=1)*self.weight_scale# 0 ~ (max_value - min_value)
        t2v_weight.fill_diagonal_(1)
        t2v_log_sm = F.log_softmax(logits*t2v_weight, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_weight = F.normalize(weight,dim=0)*self.weight_scale
        v2t_weight.fill_diagonal_(1)
        v2t_log_sm = F.log_softmax(logits*v2t_weight, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0
  
class Retrieval_Regression_Loss(torch.nn.Module):

    def __init__(self):
        super(Retrieval_Regression_Loss, self).__init__()
        self.ce_loss_func = torch.nn.CrossEntropyLoss()
        self.kl_loss_func = torch.nn.KLDivLoss(reduction="sum")
        self.reg_loss_func = torch.nn.L1Loss()

    def forward(self, sims, pred, target, label):
        '''
        logits: sim between every sample and each class
        label: the GT class
        pred/target: pred/gt icp value
        '''
        losses = {}

        label = label.squeeze()
        
        losses["ce_loss"] = self.compute_ce_dis_loss(sims, label, 4)
        losses["kl_loss"] = self.compute_kl_dis_loss(sims, label, 4)
        losses["reg_loss"] = self.reg_loss_func(pred.view(target.shape), target)

        return losses

    def compute_kl_dis_loss(self, logits, y, d):
        y_t = F.one_hot(y, d).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0
        logits_t = logits.t()

        y_column = y.T
        y_column = torch.unsqueeze(y_column,1)

        ls_weight = []

        for i in range(logits_t.shape[0]):
            if y_t_row_ind[i] > 0:
                label_inv_ranks = (torch.abs(i - y_column).transpose(0,1))
                label_inv_ranks_norm = (torch.abs(i - y_column).transpose(0,1)) / torch.sum(label_inv_ranks,dim=1) * (d-1)
                label_inv_ranks_norm = torch.squeeze(label_inv_ranks_norm,0)
                label_inv_ranks_norm[y_t[i]==1] = 1.0
                ls_label_inv_ranks_norm = label_inv_ranks_norm.detach().cpu().numpy().tolist()

            else:
                label_inv_ranks_norm = torch.ones(logits_t.shape[1]).to('cuda:0')
                ls_label_inv_ranks_norm = label_inv_ranks_norm.detach().cpu().numpy().tolist()
            
            ls_weight.append(ls_label_inv_ranks_norm)

        weight = torch.Tensor(ls_weight).to('cuda:0')
        logits_weight = logits_t * weight

        kl_loss = self.kl_loss_func(F.log_softmax(logits_weight, dim=-1), y_t_reduction) / num_slots
        return kl_loss


    def compute_ce_dis_loss(self,logits,y,d):

        list_target = list(range(d))
        target = torch.Tensor(list_target).to('cuda:0')
        target = torch.unsqueeze(target,1)

        ls_weight = []

        for i in range(len(y)):
            label_inv_ranks = (torch.abs(y[i] - target).transpose(0,1))
            label_inv_ranks_norm = (torch.abs(y[i] - target).transpose(0,1)) / torch.sum(label_inv_ranks,dim=1) * (d-1)
            label_inv_ranks_norm = torch.squeeze(label_inv_ranks_norm,0)
            label_inv_ranks_norm[y[i]] = 1.0
            ls_label_inv_ranks_norm = label_inv_ranks_norm.detach().cpu().numpy().tolist()
            ls_weight.append(ls_label_inv_ranks_norm)

        weight = torch.Tensor(ls_weight).to('cuda:0') #[B,d], indicates the distance of every sample to each class

        logits_weight = logits * weight
        loss = self.ce_loss_func(logits_weight, y)

        return loss
    

def weighted_mse_loss(inputs, targets, weights=None):
    loss = F.mse_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=0.5):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=20., gamma=1):
    loss = F.mse_loss(inputs, targets, reduce=False)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=20., gamma=1):
    loss = F.l1_loss(inputs, targets, reduce=False)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss



import torch
import torch.nn.functional as F

# --------- DDP-friendly gather（对本地输入保梯度）---------
def _gather_with_grad(x):
    """
    x: (B, D) or (B,) on each rank
    return: concat across ranks with grad for local x
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return x
    try:
        from torch.distributed.nn.functional import all_gather as all_gather_with_grad
        parts = all_gather_with_grad(x)
        return torch.cat(parts, dim=0)
    except Exception:
        # fallback: 对远端样本不保梯度（通常足够）
        world = torch.distributed.get_world_size()
        buf = [torch.zeros_like(x) for _ in range(world)]
        torch.distributed.all_gather(buf, x.contiguous())
        return torch.cat(buf, dim=0)

# --------- 软排名（Spearman 的可微近似）---------
def _soft_rank(v, tau=0.05):
    """
    v: (..., N) 向量，返回同形状的软秩（越大表示秩越高）
    经典近似: rank_i ≈ 1 + sum_j sigmoid( (v_i - v_j)/tau )
    """
    # (..., N, 1) - (..., 1, N) => (..., N, N)
    diff = (v.unsqueeze(-1) - v.unsqueeze(-2)) / tau
    P = torch.sigmoid(diff)            # 近似比较矩阵
    sr = P.sum(dim=-1)                 # (..., N) 期望秩（0~N-1）
    return sr

def _pearson_corr(x, y, eps=1e-8):
    """
    x,y: (..., N)  同长度的向量
    return: (...,)  Pearson 相关
    """
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    num = (x * y).sum(dim=-1)
    den = (x.square().sum(dim=-1).sqrt() * y.square().sum(dim=-1).sqrt()).clamp_min(eps)
    return num / den

@torch.no_grad()
def _pairwise_label_dist(y):
    # y: (B,)  ->  Dy[i,j] = |y_i - y_j|
    return (y.unsqueeze(1) - y.unsqueeze(0)).abs()

def _pairwise_embed_dist(emb, use_cosine=True):
    """
    emb: (B, E)
    返回 Dz[i,j]：推荐用负余弦相似度当“距离”
    """
    if use_cosine:
        emb = F.normalize(emb, dim=-1)
        sim = emb @ emb.t()               # (B, B)
        dz = -sim                         # 负余弦当“距离”
    else:
        # 欧氏距离
        sq = (emb**2).sum(dim=1, keepdim=True)     # (B,1)
        dz = (sq - 2*emb@emb.t() + sq.t()).clamp_min_(0).sqrt()
    return dz

def spearman_rank_loss(
    img_embeddings: torch.Tensor,   # (B, E)
    icp_value: torch.Tensor,        # (B,) or (B,1)
    tau: float = 0.05,              # 软排名温度（越小越接近硬秩，但更难训）
    exclude_self: bool = True,      # 是否在每个 anchor 下排除自身
    use_ddp_gather: bool = False,   # 是否跨卡 gather 再做排名
    use_cosine: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spearman Correlation Ranking Loss:
    - 对每个 anchor i：比较“表示距离序列 Dz[i,:]”与“标签距离序列 Dy[i,:]”的秩相关系数
    - 损失为 -mean_i SpearmanCorr
    """
    z = img_embeddings
    if icp_value.ndim == 2 and icp_value.size(1) == 1:
        y = icp_value.squeeze(1)
    else:
        y = icp_value

    if use_ddp_gather and torch.distributed.is_available() and torch.distributed.is_initialized():
        # 跨卡扩充样本，提升稳定性
        z = _gather_with_grad(z)              # (B_total, E)  本地部分保梯度
        y = _gather_with_grad(y)              # (B_total,)

    B = z.size(0)
    if B <= 1:
        return z.new_tensor(0.0)

    Dz = _pairwise_embed_dist(z, use_cosine=use_cosine)   # (B, B)
    Dy = _pairwise_label_dist(y)                          # (B, B)

    if exclude_self:
        mask = torch.eye(B, dtype=torch.bool, device=z.device)
        # 把自身项置为该行最大值，避免影响秩（也可直接去掉该位置）
        Dz = Dz.masked_fill(mask, Dz.max().detach())
        Dy = Dy.masked_fill(mask, Dy.max().detach())

    # 对每个 anchor 的“距离向量”做软排名：得到秩序列 Rz[i,:], Ry[i,:]
    Rz = _soft_rank(Dz, tau=tau)    # (B, B)
    Ry = _soft_rank(Dy, tau=tau)    # (B, B)

    # 逐 anchor 的 Pearson( Rz[i,:], Ry[i,:] )
    corr = _pearson_corr(Rz, Ry, eps=eps)   # (B,)

    loss = (1-corr).mean()   # 负相关作为损失
    return loss


import torch.distributed as dist

def huber_ccc_loss(pred, tgt, delta=1.0, eps=1e-8, sync_ddp=True):
    # --- Huber loss (本地即可) ---
    huber = torch.nn.HuberLoss(delta=delta)(pred, tgt)

    # --- Flatten & local统计 ---
    pred = pred.view(-1)
    tgt  = tgt.view(-1)
    vx = pred.var(unbiased=False)
    vy = tgt.var(unbiased=False)
    mx = pred.mean()
    my = tgt.mean()
    cov = ((pred - mx) * (tgt - my)).mean()

    if sync_ddp and dist.is_initialized():
        # 聚合每个GPU的均值、方差、协方差（乘样本数再平均）
        n = torch.tensor([len(pred)], device=pred.device, dtype=torch.float)
        stats = torch.stack([mx * n, my * n, vx * n, vy * n, cov * n, n])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        mx, my, vx, vy, cov, n = stats / stats[-1]
    ccc = (2 * cov) / (vx + vy + (mx - my).pow(2) + eps)

    return huber + (1.0 - ccc)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class HuberCCCLoss(nn.Module):
    """
    L = w_huber * Huber(delta) + w_ccc * (1 - CCC)
    - DDP: 通过 all_reduce 同步 ∑x, ∑y, ∑x², ∑y², ∑xy, n，得到真正的全局 mean/var/cov
    - huber 的 reduction 固定为 'mean'
    """
    def __init__(self, delta: float = 1.0, eps: float = 1e-8,
                 sync_ddp: bool = True, w_huber: float = 1.0, w_ccc: float = 1.0):
        super().__init__()
        self.delta = float(delta)
        self.eps = float(eps)
        self.sync_ddp = bool(sync_ddp)
        self.w_huber = float(w_huber)
        self.w_ccc = float(w_ccc)

    @staticmethod
    def _ddp_is_on():
        return dist.is_available() and dist.is_initialized()

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        huber = F.huber_loss(pred, tgt, delta=self.delta, reduction='mean')

        x = pred.view(-1)
        y = tgt.view(-1)
        if x.numel() == 0:
            # 空 batch（某些分布式采样边界会发生），直接返回 0 loss
            return x.new_tensor(0.0, dtype=pred.dtype, requires_grad=True)

        with torch.no_grad():
            x64 = x.detach().to(torch.float64)
            y64 = y.detach().to(torch.float64)

            sx  = x64.sum()                    # shape: []
            sy  = y64.sum()                    # []
            sx2 = (x64 * x64).sum()            # []
            sy2 = (y64 * y64).sum()            # []
            sxy = (x64 * y64).sum()            # []
            n   = x64.new_tensor(x64.numel(), dtype=torch.float64)  # 注意：标量张量（去掉方括号）

            stats = torch.stack([sx, sy, sx2, sy2, sxy, n])         # 全是 0 维，能 stack

            if self.sync_ddp and self._ddp_is_on():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            sx, sy, sx2, sy2, sxy, n = stats
            n = n.clamp(min=1.0)

            mx  = sx / n
            my  = sy / n
            ex2 = sx2 / n
            ey2 = sy2 / n
            exy = sxy / n

            vx  = (ex2 - mx * mx).clamp_min(0.0)
            vy  = (ey2 - my * my).clamp_min(0.0)
            cov =  exy - mx * my

        # cast 回计算 dtype
        dtype, device = x.dtype, x.device
        mx, my, vx, vy, cov = [t.to(device=device, dtype=dtype) for t in (mx, my, vx, vy, cov)]

        ccc = (2.0 * cov) / (vx + vy + (mx - my).pow(2) + self.eps)
        return self.w_huber * huber + self.w_ccc * (1.0 - ccc)
