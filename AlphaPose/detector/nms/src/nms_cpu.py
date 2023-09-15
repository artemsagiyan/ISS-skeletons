import torch


def nms_cpu_kernel(dets, threshold):
    if dets.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    x1_t = dets[:, 0].contiguous()
    y1_t = dets[:, 1].contiguous()
    x2_t = dets[:, 2].contiguous()
    y2_t = dets[:, 3].contiguous()
    scores = dets[:, 4].contiguous()

    areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)

    _, order_t = scores.sort(0, descending=True)

    ndets = dets.size(0)
    suppressed = torch.zeros(ndets, dtype=torch.uint8)

    order = order_t.data
    x1 = x1_t.data
    y1 = y1_t.data
    x2 = x2_t.data
    y2 = y2_t.data
    areas = areas_t.data

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= threshold:
                suppressed[j] = 1

    return (suppressed == 0).nonzero().squeeze(1)

def nms(dets, threshold):
    return nms_cpu_kernel(dets, threshold)