import torch


def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T=3,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = torch.nn.functional.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
            T * T)

    return kd_loss
