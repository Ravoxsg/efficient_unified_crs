import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# set seed to ensure exact reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# count learnable parameters
def count_parameters(logger, tokenizer, dec_model, model, param_optimizer, args):
    total_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("\n")
    s = "Total params"
    logger.info(f"{s:<40s}: {total_count}, trainable params: {trainable_count}, or {100*trainable_count/total_count:.4f}%, len of param_optimizer: {len(param_optimizer)}")
    embeds_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("wte.weight" in n))

    if args.only_tune_new_tokens:
        embeds_params_count -= (len(tokenizer) - args.original_token_emb_size) * dec_model.config.n_embd
    transformer_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("transformer" in n) and not ("wte.weight" in n) and not ("lora" in n))
    projection_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("projection" in n))
    lora_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("lora" in n))
    recall_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("recall" in n))
    rerank_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("rerank" in n))
    item_head_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("item_head" in n))
    weights_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("weights" in n))
    s = "LM/WTE trainable params"
    logger.info(f"{s:<40s}: {embeds_params_count} or {(100 * embeds_params_count / trainable_count):.4f}% of trainable")
    s = "LM/Transformer trainable params"
    logger.info(f"{s:<40s}: {transformer_params_count} or {(100 * transformer_params_count / trainable_count):.4f} % of trainable", main_process_only=True)
    s = "LM/Projection trainable params"
    logger.info(f"{s:<40s}: {projection_params_count} or {(100 * projection_params_count / trainable_count):.4f} % of trainable", main_process_only=True)
    s = "LM/Lora trainable params"
    logger.info(f"{s:<40s}: {lora_params_count} or {(100 * lora_params_count / trainable_count):.4f}% of trainable", main_process_only=True)
    s = "Item/recall trainable params"
    logger.info(f"{s:<40s}: {recall_params_count} or {(100 * recall_params_count / trainable_count):.4f}% of trainable", main_process_only=True)
    s = "Item/rerank trainable params"
    logger.info(f"{s:<40s}: {rerank_params_count} or {(100 * rerank_params_count / trainable_count):.4f}% of trainable", main_process_only=True)
    s = "Item/item head trainable params"
    logger.info(f"{s:<40s}: {item_head_params_count} or {(100 * item_head_params_count / trainable_count):.4f}% of trainable", main_process_only=True)
    s = "Item/weights trainable params"
    logger.info(f"{s:<40s}: {weights_params_count} or {(100 * weights_params_count / trainable_count):.4f}% of trainable", main_process_only=True)
    s = "Total trainable params"
    total_count = embeds_params_count + transformer_params_count + projection_params_count + \
                  lora_params_count + recall_params_count + rerank_params_count + item_head_params_count + \
                  weights_params_count
    logger.info(f"{s:<40s}: {total_count}", main_process_only=True)
    logger.info("\n")

# loss function
class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)

def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    """
    label_smoothing : `float`, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=-1, index=targets_flat)

    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])

    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce:
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        if reduce == "batch":
            # shape : scalar
            loss = loss.mean()

    return loss

# sample negative items (by IDs), used during training the recommendation task
def sample_ids_from_db(gt_id, num_samples, args, previous_recommended_ids=None):
    ids_2_sample_from = list(args.items_db.keys())
    ids_2_sample_from.remove(gt_id)
    results = random.sample(ids_2_sample_from, num_samples-1)
    if previous_recommended_ids is not None:
        extra_ids = previous_recommended_ids
        extra_ids = [x for x in extra_ids if x != gt_id]
        args.previous_count.append(len(extra_ids))
        results += extra_ids
    results = results[-(num_samples-1):]
    results.append(gt_id)

    return results