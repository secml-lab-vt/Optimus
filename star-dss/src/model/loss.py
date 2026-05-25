import torch
from openrlhf.models.utils import compute_approx_kl, log_probs_from_logits
from torch import Tensor, nn

__all__ = ["ValueWeightedGPTLMLoss"]


class ValueWeightedGPTLMLoss(nn.Module):
    def __init__(
        self,
        use_value: bool = False,
        use_kl: bool = False,
        ring_attn_group=None,
        kl_estimator: str = "k2",
        kl_scale: float = 1.0,
    ):
        """
        will add kl loss in the future
        """
        super().__init__()
        self.IGNORE_INDEX = -100
        self.ce_loss_fn = nn.CrossEntropyLoss(
            reduction="none" if use_value else "mean", ignore_index=self.IGNORE_INDEX
        )
        self.use_value = use_value
        self.use_kl = use_kl
        self.kl_estimator = kl_estimator
        self.kl_scale = kl_scale

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            raise NotImplementedError

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        values: Tensor = None,
        positions: Tensor = None,
        ref_logits: Tensor = None,
    ):
        """
        Compute value-weighted CE loss.

        Args:
            logits: (B, S, V)
            labels: (B, S)
            values: (B, num_chunks)
            M: chunk size
        Returns:
            Scalar loss
        """
        B, S, V = logits.shape

        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # B, S, V = logits.shape

        # conventional CE loss
        if not self.use_value:
            if self.use_kl:
                ref_logits = ref_logits[..., :-1, :].contiguous()
                per_token_logps = log_probs_from_logits(logits, labels)
                ref_per_token_logps = log_probs_from_logits(ref_logits, labels)
                kl_loss = compute_approx_kl(
                    log_probs=per_token_logps,
                    log_probs_base=ref_per_token_logps,
                    # action_mask=labels,
                    kl_estimator=self.kl_estimator,
                )

                loss = kl_loss.sum() / labels[labels != self.IGNORE_INDEX].bool().sum()

                # print(kl_loss.sum(), labels[labels != self.IGNORE_INDEX].bool().sum())

                return (loss,)
            else:
                loss = self.ce_loss_fn(logits.view(-1, V), labels.view(-1))
                # print(logits.view(-1, V).shape)
                return (loss,)

        # use value - RS (no kl) weights sum of CE and KL
        else:
            # 1. Compute per-token CE loss (shape: B, S)
            ce_loss = self.ce_loss_fn(logits.permute(0, 2, 1), labels)  # shape: (B, S)

            if self.use_kl:
                # Compute per-token KL loss (shape: B, S)
                ref_logits = ref_logits[..., :-1, :].contiguous()
                per_token_logps = log_probs_from_logits(logits, labels)
                ref_per_token_logps = log_probs_from_logits(ref_logits, labels)
                kl_loss = compute_approx_kl(
                    log_probs=per_token_logps,
                    log_probs_base=ref_per_token_logps,
                    # action_mask=labels,
                    kl_estimator=self.kl_estimator,
                )

            # print(ce_loss.shape, kl_loss.shape)

            # 2. Build token-level weights from value function
            token_weights = self.build_token_weights(
                values=values,
                positions=positions,
                seq_len=S,
            )  # shape: (B, S)
            
            # RS
            if token_weights.shape[-1] == 1:
                token_weights = token_weights.to(logits.device)
            # value + position
            else:
                token_weights = token_weights[:, 1:].to(logits.device)
            
            # print(token_weights.shape)
            # print(token_weights[0], ce_loss[0], kl_loss[0])

            # 3. Mask out ignore_index tokens
            valid_mask = (labels != self.IGNORE_INDEX).float()  # shape: (B, S-1)

            # 4. Apply weights
            weighted_ce = ce_loss * token_weights
            if self.use_kl:
                weighted_kl = kl_loss * (1 - token_weights) * self.kl_scale
                weighted_loss = weighted_ce + weighted_kl  # shape: (B, S-1)
            else:
                weighted_loss = weighted_ce
                
            ## if flash attn CE doesn't work, we need the following logic to manually mask out the prompt and the padding token 
            # weighted_loss = weighted_loss * valid_mask
            
            # return ce_loss, kl_loss

            # print(ce_loss[1], kl_loss[1], weighted_loss[1])

            # 5. Normalize over valid tokens
            final_loss = weighted_loss.sum() / valid_mask.sum()

            return (
                final_loss,
                weighted_ce.sum() / valid_mask.sum(),
                weighted_kl.sum() / valid_mask.sum() if self.use_kl else None,
            )

    def build_token_weights(self, values: Tensor, positions: Tensor, seq_len: int):
        """
        Vectorized per-token weight builder based on chunk positions and values.
        Args:
            values: Tensor (B, num_chunks)
            positions: Tensor (B, num_chunks)
            seq_len: int
        Returns:
            token_weights: Tensor (B, seq_len)
        """

        B, num_chunks = values.shape

        # Case 1: single value per sequence (B,) for Rejection Sampling
        if num_chunks == 1:
            return values

        # Case 2: chunked values (B, num_chunks)
        # Mask valid positions (positions == -1 means padding)
        valid_mask = positions != -1  # (B, num_chunks)

        # Ensure positions are cumulative boundaries
        # Add dummy "end position" for easy diff
        positions_with_end = torch.cat(
            [
                torch.full((B, 1), 0, dtype=positions.dtype),
                positions,
                # torch.full((B, 1), seq_len, dtype=positions.dtype),
            ],
            dim=1,
        )  # shape: (B, num_chunks + 1)

        # Compute spans: [start_idx, end_idx) for each chunk
        span_starts = torch.where(valid_mask, positions_with_end[:, :-1], 0)
        span_ends = torch.where(valid_mask, positions_with_end[:, 1:], 0)

        # Prepare token range
        token_range = torch.arange(seq_len).view(
            1, 1, seq_len
        )  # shape: (1, 1, seq_len)

        # Broadcast spans shape: (B, num_chunks, seq_len)
        span_mask = (token_range >= span_starts.unsqueeze(-1)) & (
            token_range < span_ends.unsqueeze(-1)
        )

        # Multiply mask by values
        weighted_spans = span_mask * values.unsqueeze(
            -1
        )  # shape: (B, num_chunks, seq_len)

        # Sum over chunks
        token_weights = weighted_spans.sum(dim=1)  # shape: (B, seq_len)

        return token_weights

    # # Case 1: single value per sequence (B,)
    # if T == 1:
    #     return values.to(device)
    #     # alpha = values
    #     # # alpha = sigmoid_weight(values, k=k, tau=tau)  # (B,)
    #     # return (
    #     #     alpha[:, None].expand(B, seq_len).to(device)
    #     # )  # broadcast to all tokens

    # # Case 2: chunked values (B, num_chunks)
    # else:
    #     raise NotImplementedError
    # # num_chunks = values.shape[1]
    # token_weights = torch.zeros(B, seq_len, device=device)
    # for i in range(num_chunks):
    #     alpha_i = sigmoid_weight(values[:, i], k=k, tau=tau)  # (B,)
    #     start = i * M
    #     end = min((i + 1) * M, seq_len)
    #     token_weights[:, start:end] = alpha_i[
    #         :, None
    #     ]  # broadcast alpha to each chunk

    # return token_weights

    # def sigmoid_weight(self, values, k=5.0, tau=0.5):
    #     """Maps value function output to alpha (weight on CE loss)"""
    #     return 1 / (1 + torch.exp(-k * (values - tau)))  # shape: (B, num_chunks)
