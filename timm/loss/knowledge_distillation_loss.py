import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CRDLoss(nn.Module):
    """ Contrastive Representation Distillation (CRD)
        在特征空间中最大化正样本对的相似性，同时最小化负样本对的相似性。
        正样本对指的是：学生和教师模型对同一张图片的特征。我们希望正样本对之间的相似度尽量高。
        负样本对指的是：学生和教师模型对不同图片的特征。我们希望负样本对之间的相似度尽量低。
    """
    def __init__(self, temperature=0.5):
        super(CRDLoss, self).__init__()
        self.temperature = temperature
    
    """
        teacher_features = teacher_model(images)
        student_features = student_model(images)
        positive_indices = torch.arange(images.size(0))  # 假设正样本是batch内的顺序对应
    """
    def forward(self, student_features, teacher_features, positive_indices):
        # Normalizing features
        student_features = F.normalize(student_features, dim=1)
        teacher_features = F.normalize(teacher_features, dim=1)

        # Calculate similarities
        similarity_matrix = torch.mm(student_features, teacher_features.t())
        logits = similarity_matrix / self.temperature

        # Generate positive pairs using given indices
        batch_size = student_features.size(0)
        labels = torch.arange(batch_size).to(student_features.device)
        labels = labels[positive_indices]

        # Compute InfoNCE loss
        # 计算每一行的交叉熵损失 
        # 交叉熵损失会试图将 logits 的每一行中对应标签的值最大化（即让正样本对的相似度最大），同时减小其他位置的值（负样本对的相似度），从而实现对比学习的目标。
        # logits 的每一行作为预测的相似度分布。
        # labels 的每个值作为每行的目标类（正样本对的标签）。
        loss = F.cross_entropy(logits, labels)
        return loss


class DirectNormLoss(nn.Module):
    '''
    它可以帮助学生模型(student)学习教师模型(teacher)的类间和类内特征分布结构。
    这种应用的主要目的是增强学生模型的表达能力，使其更接近教师模型在特征空间的分布。
    '''
    def __init__(self, num_class=100):
        super(DirectNormLoss, self).__init__()
        self.num_class = num_class
        # self.nd_loss_factor = nd_loss_factor
    
    def project_center(self, s_emb, t_emb, T_EMB, labels):
        assert s_emb.size() == t_emb.size()
        assert s_emb.shape[0] == len(labels)
        loss = 0.0
        for s, t, i in zip(s_emb, t_emb, labels):
            i = i.item()
            center = torch.tensor(T_EMB[str(i)]).cuda()
            e_c = center / center.norm(p=2)
            max_norm = max(s.norm(p=2), t.norm(p=2))
            loss += 1 - torch.dot(s, e_c) / max_norm
        return loss
     
    def forward(self, s_emb, t_emb, T_EMB, labels):
        nd_loss = self.project_center(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels)
        
        return nd_loss / len(labels)
