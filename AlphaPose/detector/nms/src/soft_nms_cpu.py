import numpy as np


def soft_nms_cpu(boxes_in, iou_thr, method=1, sigma=0.5, min_score=0.001):
    boxes = boxes_in.copy()
    N = boxes.shape[0]
    inds = np.arange(N)

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1, ty1, tx2, ty2, ts = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], boxes[i, 4]
        ti = inds[i]

        pos = i + 1

        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], boxes[i, 4] = \
            boxes[maxpos, 0], boxes[maxpos, 1], boxes[maxpos, 2], boxes[maxpos, 3], boxes[maxpos, 4]
        inds[i], inds[maxpos] = inds[maxpos], ti

        tx1, ty1, tx2, ty2, ts = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], boxes[i, 4]

        pos = i + 1

        while pos < N:
            x1, y1, x2, y2, s = boxes[pos, 0], boxes[pos, 1], boxes[pos, 2], boxes[pos, 3], boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = max(0, min(tx2, x2) - max(tx1, x1) + 1)

            if iw > 0:
                ih = max(0, min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua

                    if method == 1:
                        if ov > iou_thr:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:
                        weight = np.exp(-(ov * ov) / sigma)
                    else:
                        if ov > iou_thr:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    if boxes[pos, 4] < min_score:
                        boxes[pos, 0], boxes[pos, 1], boxes[pos, 2], boxes[pos, 3], boxes[pos, 4] = \
                            boxes[N-1, 0], boxes[N-1, 1], boxes[N-1, 2], boxes[N-1, 3], boxes[N-1, 4]
                        inds[pos] = inds[N - 1]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    return boxes[:N], inds[:N]