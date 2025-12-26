import torch
import torch.nn.functional as F
from ..modules.utils import get_x_2d
from einops import rearrange


def get_correspondences(R, K, T, img_h, img_w, plane_depth=1.0):
    """
    Build per-pair image correspondences grid that accounts for both rotation and translation.
    Args:
        R: [b, m, 3, 3]   rotations of each view w.r.t the same reference frame (your 'real')
        K: [b, m, 3, 3]   intrinsics per view
        T: [b, m, 3] or None
           translations of each view w.r.t the same reference frame (cond -> real).
           If None, falls back to rotation-only homography.
        img_h, img_w: output grid size (query image spatial size)
        plane_depth: scalar depth (in left camera coords) for a fronto-parallel plane (default 1.0)
    Returns:
        correspondences: [b, m, m, img_h, img_w, 2]
            For each pair (i -> j), a grid of target j pixel coords corresponding to pixels in i.
    """
    b, m = R.shape[0], R.shape[1]
    device = R.device
    dtype  = R.dtype

    # Pixel mesh in homogeneous coords once, on device
    # get_x_2d returns HxWx3 with last dim [x,y,1] in pixel coords
    xy1 = torch.tensor(get_x_2d(img_w, img_h), device=device, dtype=dtype)  # [H, W, 3]
    xy1 = xy1.view(-1, 3).t().unsqueeze(0).repeat(b, 1, 1)                  # [b, 3, H*W]

    correspondences = torch.zeros((b, m, m, img_h, img_w, 2), device=device, dtype=dtype)

    invK = torch.linalg.inv(K)    # [b, m, 3, 3]
    Rt   = R.transpose(-1, -2)    # [b, m, 3, 3] since R is rotation

    n = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3, 1)  # [1,3,1]
    d = torch.tensor(plane_depth, device=device, dtype=dtype).view(1, 1, 1)      # [1,1,1]

    use_T = T is not None
    if use_T:
        T = T.to(device=device, dtype=dtype)  # [b, m, 3]

    for i in range(m):      # left/query
       
        R_left  = R[:, i]         # [b, 3, 3]
        K_left  = K[:, i]         # [b, 3, 3]
        invK_l  = invK[:, i]      # [b, 3, 3]
        Rt_left = Rt[:, i]        # [b, 3, 3]
        if use_T:
            t_left = T[:, i]      # [b, 3]

        for j in range(m):  # right/key
            R_right  = R[:, j]          # [b, 3, 3]
            K_right  = K[:, j]          # [b, 3, 3]
            invR_r   = Rt[:, j]         # [b, 3, 3] (R_right^-1)

            # Relative rotation left->right (both defined w.r.t same ref frame)
            # R_lr = R_right^-1 * R_left
            R_lr = torch.matmul(invR_r, R_left)               # [b, 3, 3]

            if use_T:
                # t_lr = R_right^-1 * (t_left - t_right)
                t_right = T[:, j]                              # [b, 3]
                t_lr = torch.matmul(invR_r, (t_left - t_right).unsqueeze(-1)).squeeze(-1)  # [b,3]
                # Plane-induced homography: H = K_r * (R_lr - (t_lr n^T)/d) * inv(K_l)
                t_nT_over_d = torch.matmul(t_lr.unsqueeze(-1), n.transpose(1, 2)) / d  # [b,3,3]
                H_core = R_lr - t_nT_over_d                                            # [b,3,3]
            else:
                # Rotation-only homography (old behavior): H = K_r * (R_r^-1 * R_l) * inv(K_l)
                H_core = R_lr

            H = torch.matmul(K_right, torch.matmul(H_core, invK_l))  # [b, 3, 3]

            # Apply H to all pixels of left view
            xyz_r = torch.matmul(H, xy1)                 # [b, 3, H*W]
            xy_r  = (xyz_r[:, :2] / (xyz_r[:, 2:] + 1e-8))  # [b, 2, H*W]
            xy_r  = xy_r.permute(0, 2, 1).view(b, img_h, img_w, 2)   # [b, H, W, 2]

            correspondences[:, i, j] = xy_r

    return correspondences



def get_key_value(key_value, xy_l, homo_r, ori_h, ori_w, ori_h_r, query_h):
    
    b, c, h, w = key_value.shape
    query_scale = ori_h//query_h
    key_scale = ori_h_r//h

    xy_l = xy_l[:, query_scale//2::query_scale,
                query_scale//2::query_scale]/key_scale-0.5

    key_values = []

    xy_proj = []
    kernal_size=3
    for i in range(0-kernal_size//2, 1+kernal_size//2):
        for j in range(0-kernal_size//2, 1+kernal_size//2):
            xy_l_norm = xy_l.clone()
            xy_l_norm[..., 0] = xy_l_norm[..., 0] + i
            xy_l_norm[..., 1] = xy_l_norm[..., 1] + j
            xy_l_rescale = (xy_l_norm+0.5)*key_scale

            xy_proj.append(xy_l_rescale)

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            _key_value = F.grid_sample(
                key_value, xy_l_norm, align_corners=True)
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    mask = (xy_proj[..., 0] > 0)*(xy_proj[..., 0] < ori_w) * \
        (xy_proj[..., 1] > 0)*(xy_proj[..., 1] < ori_h)

    xy_proj_back = torch.cat([xy_proj, torch.ones(
        *xy_proj.shape[:-1], 1, device=xy_proj.device)], dim=-1)
    xy_proj_back = rearrange(xy_proj_back, 'b n h w c -> b c (n h w)')
    xy_proj_back = homo_r@xy_proj_back
    
    xy_proj_back = rearrange(
        xy_proj_back, 'b c (n h w) -> b n h w c', h=h, w=w)
    xy_proj_back = xy_proj_back[..., :2]/xy_proj_back[..., 2:]

    xy = get_x_2d(ori_w, ori_h)[:, :, :2]
    xy = xy[query_scale//2::query_scale, query_scale//2::query_scale]
    xy = torch.tensor(xy, device=key_value.device).float()[
        None, None]

    xy_rel = (xy_proj_back-xy)/query_scale

    key_values = torch.stack(key_values, dim=1)

    return key_values, xy_rel, mask


def get_query_value(query, key_value, xy_l, homo_r, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
    if img_h_r is None:
        img_h_r = img_h_l
        img_w_r = img_w_l

    b = query.shape[0]
    m = key_value.shape[1]

    key_values = []
    masks = []
    xys = []

    for i in range(m):
        _, _, q_h, q_w = query.shape
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], homo_r[:, i],
                                               img_h_l, img_w_l, img_w_r, q_h)

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask
