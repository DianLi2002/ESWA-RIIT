import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_sam=1.0):
        super(PhysicsInformedLoss, self).__init__()

        self.lambda_sam = lambda_sam 
        self.main_loss_fn = CharbonnierLoss() 
        self.sam_loss = SAMLoss()  
    def forward(self, pred_R, gt_R):
        # 1. Main Task (Reflectance) - 用 Charbonnier
        loss_main = self.main_loss_fn(pred_R, gt_R)
        # 2. SAM Loss
        loss_sam = self.sam_loss(pred_R, gt_R)
        total_loss = loss_main + self.lambda_sam * loss_sam
        return total_loss
    
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, x, y):
        # x: pred_R [B, C, H, W]
        # y: gt_R   [B, C, H, W]
        
        # 1. 计算点积 (Dot Product) -> [B, 1, H, W]
        dot_sum = torch.sum(x * y, dim=1, keepdim=True)
        
        # 2. 计算模长 (Norm)
        norm_x = torch.norm(x, dim=1, keepdim=True)
        norm_y = torch.norm(y, dim=1, keepdim=True)
        
        # 3. 计算 Cosine Similarity
        # 加一个极小值 eps 防止除零
        cos_sim = dot_sum / (norm_x * norm_y + 1e-8)
        
        # 4. 截断 (Clamp) 防止 acos 出界 (数值稳定性)
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        
        # 5. 计算角度 (Arccos)
        sam = torch.acos(cos_sim)
        
        # 6. 平均
        return torch.mean(sam)