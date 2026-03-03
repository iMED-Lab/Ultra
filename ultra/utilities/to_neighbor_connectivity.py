# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def bresenham_line(dx, dy):
    x0, y0 = 0, 0
    x1, y1 = dx, dy
    abs_dx, abs_dy = abs(x1), abs(y1)
    sx = 1 if x1 > 0 else -1
    sy = 1 if y1 > 0 else -1
    offsets = []
    if abs_dx >= abs_dy:
        err = abs_dx // 2
        x, y = x0, y0
        while (x, y) != (x1, y1):
            x += sx
            err -= abs_dy
            if err < 0:
                y += sy
                err += abs_dx
            offsets.append((x, y))
    else:
        err = abs_dy // 2
        x, y = x0, y0
        while (x, y) != (x1, y1):
            y += sy
            err -= abs_dx
            if err < 0:
                x += sx
                err += abs_dy
            offsets.append((x, y))
    return offsets


def nk_encode(img, kernel_size=7):
    """
    Neighborhood connectivity encoding under kernel_size K
    A larger neighborhood (e.g., 7 × 7) connectivity coding is performed on the image,
    and all intermediate pixels on the path from the center to the far end are considered
    to be the same as the center pixel in the determination.
    Args:
        img: Tensor, (B, H, W) or (B, 1, H, W)
        kernel_size: size of neighborhood region, default is 7x7
    Returns:
        connectivity_maps: Tensor, [B, (kernel_size^2-1), H*W]
        vessel_mask: Tensor, [B, (kernel_size^2-1), H*W], 1 for vessel pixels, 0 for background
    """
    if img.dim() == 4 and img.size(1) == 1:
        img = img.squeeze(1)
    B, H, W = img.shape
    r = kernel_size // 2

    offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if not (dx == 0 and dy == 0)]
    num_offsets = len(offsets)
    # padding the image
    padded_img = F.pad(img.unsqueeze(1), (r, r, r, r), mode='constant', value=-1).squeeze(1)
    connectivity_maps = torch.zeros((B, H * W, num_offsets), device=img.device, dtype=torch.float32)
    # process each offset
    for k, (dx, dy) in enumerate(offsets):
        shifted_img = padded_img[:, r + dx: r + dx + H, r + dy: r + dy + W]
        endpoint_equal = (shifted_img == img)
        path_offsets = bresenham_line(dx, dy)
        if len(path_offsets) > 1:
            path_continuity = torch.ones_like(img, dtype=torch.bool)
            for (px, py) in path_offsets[:-1]:
                path_pixel = padded_img[:, r + px: r + px + H, r + py: r + py + W]
                path_continuity &= (path_pixel == img)
            connectivity_valid = endpoint_equal & path_continuity
        else:
            connectivity_valid = endpoint_equal
        connectivity_maps[:, :, k] = connectivity_valid.view(B, -1).to(torch.float32)
    connectivity_maps = connectivity_maps.permute(0, 2, 1)  # [B, N_neighbor, H*W]
    # ### Update: construct the vessel mask.
    vessel_mask = (img > 0).float().view(B, 1, -1)  # [B, 1, H*W]
    vessel_mask = vessel_mask.expand(-1, num_offsets, -1)  # [B, N_neighbor, H*W]

    return connectivity_maps, vessel_mask


def to_nk_maps(img, kernel_sizes=[3, 5, 7], masking=True):
    if isinstance(img, list):
        nk_maps, masks = [], []
        for i in img:
            if len(i.shape) == 4 and i.shape[1] == 1:
                i = i.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            nk, m = [], []
            for k in kernel_sizes:
                nk_k, m_k = nk_encode(i, kernel_size=k)
                nk.append(nk_k)
                m.append(m_k)
            # To facilitate subsequent loss calculations, we concatenate the K dimensions.
            nk = torch.cat(nk, dim=1)
            m = torch.cat(m, dim=1)
            nk_maps.append(nk)
            masks.append(m)
        if masking:
            return nk_maps, masks
        return nk_maps
    else:
        nk_maps, masks = [], []
        if len(img.shape) == 4 and img.shape[1] == 1:
            img = img.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        for k in kernel_sizes:
            nk_k, m_k = nk_encode(img, kernel_size=k)
            nk_maps.append(nk_k)
            masks.append(m_k)
        nk_maps = torch.cat(nk_maps, dim=1)
        masks = torch.cat(masks, dim=1)
        if masking:
            return nk_maps, masks
        return nk_maps


if __name__ == '__main__':
    img = torch.tensor([[[[1, 1, 2, 0, 0, 0, 0],
                          [0, 1, 2, 0, 0, 0, 0],
                          [2, 2, 2, 2, 0, 2, 0],
                          [1, 0, 2, 2, 0, 0, 0],
                          [2, 0, 0, 2, 0, 0, 2],
                          [2, 0, 0, 0, 0, 0, 0],
                          [2, 0, 0, 0, 0, 0, 0]]],
                        [[[1, 1, 2, 0, 0, 0, 0],
                          [0, 1, 2, 0, 0, 0, 0],
                          [2, 2, 2, 2, 0, 2, 0],
                          [1, 0, 2, 2, 0, 0, 0],
                          [2, 0, 0, 2, 0, 0, 2],
                          [2, 0, 0, 0, 0, 0, 0],
                          [2, 0, 0, 0, 0, 0, 0]]]
                        ])
    maps, masks = to_nk_maps(img, kernel_sizes=[3, 5, 7])  # [0]
    print(masks.shape)
    print(masks[0][:8, 3])
    print(maps.shape)
    print(maps[0][:8, 3])
