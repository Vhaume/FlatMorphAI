"""
Fixed Custom Mask2Former Losses - Handles class_labels as list correctly
=========================================================================

This fixes the TypeError when class_labels is a list of tensors (standard format).

The issue was:
    target_classes[batch_idx, src_idx] = class_labels[batch_idx, tgt_idx]
    
Should be:
    target_classes[batch_idx, src_idx] = class_labels[batch_idx][tgt_idx]

Reference:
    HuggingFace Mask2Former implementation
    
Author: Fixed implementation
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: int) -> torch.Tensor:
    """
    Compute the DICE loss, similar to generalized IOU for masks
    
    Args:
        inputs: A float tensor of arbitrary shape (predictions)
        targets: A float tensor with the same shape as inputs (ground truth)
        num_masks: Number of masks
        
    Returns:
        dice_loss: Scalar loss
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: int) -> torch.Tensor:
    """
    Compute sigmoid cross entropy loss for masks
    
    Args:
        inputs: A float tensor of arbitrary shape (predictions)
        targets: A float tensor with the same shape as inputs (ground truth)
        num_masks: Number of masks
        
    Returns:
        ce_loss: Scalar loss
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


class CustomMask2FormerCriterion(nn.Module):
    """
    FIXED Custom Mask2Former criterion that properly handles class_labels as list
    
    Changes from original:
    1. Fixed class_labels indexing: class_labels[batch_idx][tgt_idx] instead of class_labels[batch_idx, tgt_idx]
    2. Added proper handling for list format
    3. Added safety checks
    
    Args:
        num_classes (int): Number of classes
        matcher: Matcher for Hungarian matching
        class_weight (float): Weight for classification loss
        mask_weight (float): Weight for mask loss
        dice_weight (float): Weight for dice loss
        no_object_weight (float): Weight for no-object class
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        no_object_weight: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.no_object_weight = no_object_weight
        
        # Add weight_dict for compatibility with standard Mask2Former
        # This allows the model to know the relative importance of each loss
        self.weight_dict = {
            'loss_class': class_weight,
            'loss_mask': mask_weight,
            'loss_dice': dice_weight,
        }
        
        print(f"\n{'='*70}")
        print(f"CUSTOM MASK2FORMER CRITERION (FIXED)")
        print(f"{'='*70}")
        print(f"  Num classes: {num_classes}")
        print(f"  Class weight: {class_weight}")
        print(f"  Mask weight: {mask_weight}")
        print(f"  Dice weight: {dice_weight}")
        print(f"  No-object weight: {no_object_weight}")
        print(f"{'='*70}\n")
    
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        auxiliary_predictions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing all losses
        
        Args:
            masks_queries_logits: [batch_size, num_queries, height, width]
            class_queries_logits: [batch_size, num_queries, num_classes]
            mask_labels: List of [num_instances, height, width] tensors
            class_labels: List of [num_instances] tensors
            auxiliary_predictions: Optional auxiliary predictions
            
        Returns:
            loss_dict: Dictionary with all losses
        """
        device = masks_queries_logits.device
        
        # Hungarian matching
        indices = self.matcher(
            masks_queries_logits,
            class_queries_logits,
            mask_labels,
            class_labels
        )
        
        loss_dict = {}
        
        # 1. Classification Loss (FIXED)
        class_loss = self._compute_class_loss_fixed(
            class_queries_logits, class_labels, indices, device
        )
        loss_dict['loss_class'] = class_loss  # Unweighted - weight applied via weight_dict
        
        # 2. Mask Loss
        mask_loss, dice_loss = self._compute_mask_losses(
            masks_queries_logits, mask_labels, indices, device
        )
        loss_dict['loss_mask'] = mask_loss  # Unweighted
        loss_dict['loss_dice'] = dice_loss  # Unweighted
        
        # 3. Auxiliary losses (if provided)
        if auxiliary_predictions is not None:
            for i, aux_predictions in enumerate(auxiliary_predictions):
                aux_masks_logits = aux_predictions['masks_queries_logits']
                aux_class_logits = aux_predictions['class_queries_logits']
                
                # Recompute matching for auxiliary predictions
                aux_indices = self.matcher(
                    aux_masks_logits,
                    aux_class_logits,
                    mask_labels,
                    class_labels
                )
                
                # Auxiliary class loss (FIXED)
                aux_class_loss = self._compute_class_loss_fixed(
                    aux_class_logits, class_labels, aux_indices, device
                )
                loss_dict[f'loss_class_aux_{i}'] = aux_class_loss  # Unweighted
                # Add to weight_dict if not already there
                if f'loss_class_aux_{i}' not in self.weight_dict:
                    self.weight_dict[f'loss_class_aux_{i}'] = self.class_weight
                
                # Auxiliary mask losses
                aux_mask_loss, aux_dice_loss = self._compute_mask_losses(
                    aux_masks_logits, mask_labels, aux_indices, device
                )
                loss_dict[f'loss_mask_aux_{i}'] = aux_mask_loss  # Unweighted
                loss_dict[f'loss_dice_aux_{i}'] = aux_dice_loss  # Unweighted
                # Add to weight_dict if not already there
                if f'loss_mask_aux_{i}' not in self.weight_dict:
                    self.weight_dict[f'loss_mask_aux_{i}'] = self.mask_weight
                if f'loss_dice_aux_{i}' not in self.weight_dict:
                    self.weight_dict[f'loss_dice_aux_{i}'] = self.dice_weight
        
        return loss_dict
    
    def _compute_class_loss_fixed(
        self,
        class_logits: torch.Tensor,
        class_labels: List[torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device
    ) -> torch.Tensor:
        """
        FIXED: Compute classification loss with proper list handling
        
        Args:
            class_logits: [batch_size, num_queries, num_classes]
            class_labels: List of [num_instances] tensors
            indices: List of (src_idx, tgt_idx) tuples from matcher
            device: Device
            
        Returns:
            class_loss: Scalar classification loss
        """
        batch_size, num_queries, num_classes = class_logits.shape
        
        # Initialize target classes (all as no-object initially)
        target_classes = torch.full(
            (batch_size, num_queries),
            num_classes - 1,  # No-object class
            dtype=torch.long,
            device=device
        )
        
        # FIXED: Assign matched labels - handle class_labels as list
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if batch_idx < len(class_labels):
                # Extract class labels for this batch item
                batch_class_labels = class_labels[batch_idx]  # [num_instances]
                
                # Assign matched classes
                # src_idx: which queries matched [num_matches]
                # tgt_idx: which ground truth instances they matched [num_matches]
                target_classes[batch_idx, src_idx] = batch_class_labels[tgt_idx]
        
        # Compute cross entropy with no_object_weight
        class_weights = torch.ones(num_classes, device=device)
        class_weights[-1] = self.no_object_weight  # Downweight no-object class
        
        class_loss = F.cross_entropy(
            class_logits.transpose(1, 2),  # [batch_size, num_classes, num_queries]
            target_classes,                 # [batch_size, num_queries]
            weight=class_weights,
            reduction='mean'
        )
        
        return class_loss
    
    def _compute_mask_losses(
        self,
        masks_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mask losses (BCE + Dice)
        
        Args:
            masks_logits: [batch_size, num_queries, height, width]
            mask_labels: List of [num_instances, height, width] tensors
            indices: List of (src_idx, tgt_idx) tuples
            device: Device
            
        Returns:
            mask_loss: BCE loss
            dice_loss: Dice loss
        """
        # Get matched predictions and targets
        src_masks_list = []
        tgt_masks_list = []
        
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if batch_idx < len(mask_labels):
                # Get matched predictions
                src_masks = masks_logits[batch_idx, src_idx]  # [num_matches, H, W]
                
                # Get matched ground truth
                tgt_masks = mask_labels[batch_idx][tgt_idx]   # [num_matches, H, W]
                
                # Interpolate predictions to match target size
                if src_masks.shape[-2:] != tgt_masks.shape[-2:]:
                    src_masks = F.interpolate(
                        src_masks.unsqueeze(0),  # [1, num_matches, H, W]
                        size=tgt_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # [num_matches, H, W]
                
                src_masks_list.append(src_masks)
                tgt_masks_list.append(tgt_masks.float())
        
        if len(src_masks_list) == 0:
            # No matches - return zero losses
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Concatenate all matches
        src_masks = torch.cat(src_masks_list, dim=0)  # [total_matches, H, W]
        tgt_masks = torch.cat(tgt_masks_list, dim=0)  # [total_matches, H, W]
        
        num_masks = src_masks.shape[0]
        
        # Compute losses
        mask_loss = batch_sigmoid_ce_loss(src_masks, tgt_masks, num_masks)
        dice_loss = batch_dice_loss(src_masks, tgt_masks, num_masks)
        
        return mask_loss, dice_loss


class FocalBoundaryDiceLoss(nn.Module):
    """
    Focal + Boundary + Dice loss combination
    
    Reference:
        Various papers on boundary-aware segmentation
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        boundary_weight: float = 1.0,
        dice_weight: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss"""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary-aware loss using morphological operations"""
        from torch.nn.functional import max_pool2d
        
        # Compute boundaries using max_pool - erosion
        kernel_size = 3
        target_dilated = max_pool2d(target, kernel_size, stride=1, padding=kernel_size//2)
        target_eroded = -max_pool2d(-target, kernel_size, stride=1, padding=kernel_size//2)
        boundary_mask = (target_dilated - target_eroded).bool().float()
        
        # Weight loss by boundaries
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = bce * (1 + boundary_mask * self.boundary_weight)
        
        return weighted_bce.mean()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute dice loss"""
        pred = torch.sigmoid(pred)
        smooth = 1.0
        
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return 1.0 - dice
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            pred: Predictions [B, C, H, W] or [B, H, W]
            target: Targets [B, C, H, W] or [B, H, W]
        """
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total = focal + boundary * self.boundary_weight + dice * self.dice_weight
        
        return total


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_custom_criterion(
    num_classes: int,
    matcher,
    class_weight: float = 2.0,
    mask_weight: float = 5.0,
    dice_weight: float = 5.0,
    no_object_weight: float = 0.1
) -> CustomMask2FormerCriterion:
    """
    Create custom criterion with fixed class_labels handling
    
    Args:
        num_classes: Number of classes
        matcher: Hungarian matcher
        class_weight: Weight for classification loss
        mask_weight: Weight for mask BCE loss
        dice_weight: Weight for dice loss
        no_object_weight: Weight for no-object class
    
    Returns:
        criterion: Fixed custom criterion
    """
    return CustomMask2FormerCriterion(
        num_classes=num_classes,
        matcher=matcher,
        class_weight=class_weight,
        mask_weight=mask_weight,
        dice_weight=dice_weight,
        no_object_weight=no_object_weight
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example showing how to use the fixed criterion
    """
    print("\n" + "="*70)
    print("EXAMPLE: FIXED CUSTOM CRITERION")
    print("="*70 + "\n")
    
    from transformers import Mask2FormerForUniversalSegmentation
    
    # Load model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic"
    )
    
    # Create custom criterion (FIXED)
    criterion = create_custom_criterion(
        num_classes=2,  # Binary: background + tumor
        matcher=model.criterion.matcher,  # Use model's matcher
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        no_object_weight=0.1
    )
    
    # Replace model's criterion
    model.criterion = criterion
    
    print("✓ Custom criterion created and assigned")
    print("  This criterion properly handles class_labels as list")
    print("="*70 + "\n")
    
    # Test forward pass
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 512, 512)
    
    # Ground truth in standard format (LIST of tensors)
    mask_labels = [
        torch.randint(0, 2, (1, 512, 512)).float(),  # 1 instance
        torch.randint(0, 2, (2, 512, 512)).float(),  # 2 instances
    ]
    class_labels = [
        torch.tensor([1]),      # Class 1
        torch.tensor([1, 1]),   # Both class 1
    ]
    
    # Forward pass
    outputs = model(
        pixel_values=pixel_values,
        mask_labels=mask_labels,
        class_labels=class_labels
    )
    
    loss = outputs.loss
    print(f"✓ Forward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    example_usage()
