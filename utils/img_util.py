import torch
import numpy as np

def set_window_wl_ww(tensor, wl=225, ww=450, use_float=False):
    w_min, w_max = wl - ww // 2, wl + ww // 2
    if isinstance(tensor, torch.Tensor):
        tensor = torch.clip(tensor, w_min, w_max)
    else:
        tensor = np.clip(tensor, w_min, w_max)
    tensor = (1.0 * (tensor - w_min) / (w_max - w_min)) * 255.0

    if not use_float:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.uint8)
        else:
            tensor = tensor.astype(np.uint8)

    return tensor

def apply_wl_ww_and_norm(tensor, WL_WW, use_float=False):
    if WL_WW[0] is not None:
        wl, ww = WL_WW
        if isinstance(ww, tuple):
            imgs = []
            for w in ww:
                img = set_window_wl_ww(tensor, wl, w, use_float=use_float)
                imgs.append(img)
            img = torch.cat(imgs, dim=1)
        else:
            img = set_window_wl_ww(tensor, wl, ww, use_float=use_float)
    else:
        img = tensor

    # img = (img / 255.0)  # within range [0, 1]
    return img