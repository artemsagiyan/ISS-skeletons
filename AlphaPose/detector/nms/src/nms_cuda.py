import torch

from AlphaPose.detector.nms.src.nms_cpu import nms_cpu_kernel


def nms_cuda(boxes, nms_overlap_thresh):
    # Ваша реализация nms_cuda, если она доступна для PyTorch CUDA.
    pass

def nms(dets, threshold):
    if dets.is_cuda:
        return nms_cuda(dets, threshold)
    if dets.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=torch.device("cpu"))
    return nms_cpu_kernel(dets, threshold)  # Предполагая, что nms_cpu_kernel определена как в предыдущем ответе

# PyTorch extension module
class NMSModule(torch.nn.Module):
    def forward(self, dets, threshold):
        return nms(dets, threshold)

# Регистрация модуля в PyTorch
nms_module = NMSModule()