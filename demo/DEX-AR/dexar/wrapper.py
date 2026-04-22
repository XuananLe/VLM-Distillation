from dataclasses import dataclass

import torch
from einops import rearrange
from PIL import Image

from .backends import DexarBackend
from .utils import min_max, topk_norm


@dataclass
class DexarResult:
    """Output of DEX-AR explainability computation.

    Attributes:
        per_token_heatmaps: [num_tokens, H, W] filtered per-token heatmaps (Eq. 5 head filtering applied).
        per_token_heatmaps_unfiltered: [num_tokens, H, W] unfiltered per-token heatmaps (plain sum over all heads/layers).
        token_weights: [num_tokens] delta^t visual relevance weights (Eq. 6).
        sentence_heatmap: [H, W] filtered sentence-level heatmap (Eq. 6).
        sentence_heatmap_unfiltered: [H, W] unfiltered sentence-level heatmap.
        tokens: Decoded token strings.
    """
    per_token_heatmaps: torch.Tensor
    per_token_heatmaps_unfiltered: torch.Tensor
    token_weights: torch.Tensor
    sentence_heatmap: torch.Tensor
    sentence_heatmap_unfiltered: torch.Tensor
    tokens: list


class DexarWrapper:
    """Wraps a HuggingFace VLM for DEX-AR explainability.

    Args:
        model: A HuggingFace vision-language generation model.
        processor: The corresponding AutoProcessor.
        layer_index: Starting layer depth (negative = from end). Default -10.
    """

    def __init__(self, backend: DexarBackend, layer_index: int = -10):
        self.model = backend.model
        self.processor = backend.processor
        self.layer_index = layer_index
        self.backend = backend
        self.num_layers = backend.num_layers
        self.lm_head = backend.lm_head
        self.norm = backend.norm
        self.default_prompt = backend.default_prompt
        self.recommended_image_size = backend.recommended_image_size
        self.model_family = backend.family

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        layer_index: int = -10,
    ) -> "DexarWrapper":
        """Load a supported VLM and processor from HuggingFace.

        Args:
            model_name: HuggingFace model ID.
            device: Device to load model on. Default "cuda".
            layer_index: Starting layer depth. Default -10.
        """
        backend = DexarBackend.from_pretrained(model_name, device)
        backend.enable_dexar_gradients()

        return cls(backend, layer_index=layer_index)

    @torch.inference_mode(False)
    def compute_dexar(
        self,
        image: Image.Image,
        target_sentence: str,
        prompt: str = "USER: <image>\nDescribe the image. ASSISTANT:",
    ) -> DexarResult:
        """Compute DEX-AR explainability maps.

        Args:
            image: Input PIL Image.
            target_sentence: The target sentence to explain.
            prompt: Prompt template containing <image> placeholder.

        Returns:
            DexarResult with per-token heatmaps, token weights, and sentence heatmap.
        """
        device = next(self.model.parameters()).device

        # --- Tokenize prompt and target ---
        encoded_prompt = self.backend.encode_prompt(prompt=prompt, image=image, device=device)
        input_ids = encoded_prompt.model_inputs["input_ids"]
        attention_mask = encoded_prompt.model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        model_static_inputs = {
            key: value
            for key, value in encoded_prompt.model_inputs.items()
            if key not in {"input_ids", "attention_mask"}
        }
        prompt_image_mask = encoded_prompt.prompt_image_mask
        prompt_text_mask = ~prompt_image_mask
        prompt_length = encoded_prompt.prompt_input_ids.shape[1]
        h, w = encoded_prompt.image_grid

        target_ids = self.processor.tokenizer.encode(
            target_sentence, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        num_tokens_to_generate = target_ids.shape[-1]

        # Decode individual tokens for output
        token_strings = [
            self.processor.tokenizer.decode(target_ids[0, i])
            for i in range(num_tokens_to_generate)
        ]

        # --- Layer range ---
        starting_depth = (
            self.layer_index
            if self.layer_index >= 0
            else self.num_layers + self.layer_index
        )

        # --- Main token generation loop ---
        all_new_tokens_grads = []
        all_heads_topk_norm_img = []
        all_heads_topk_norm_text = []

        for n in range(num_tokens_to_generate):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_static_inputs,
            )

            next_token_id = target_ids[:, n]
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states[1:]  # skip embedding layer

            new_token_grads = []
            heads_topk_norm_img = []
            heads_topk_norm_text = []

            # Head scoring uses max mode (k=1) — Table 3 best
            k_img = 1
            k_all_text = 1

            for l in range(starting_depth, self.num_layers):
                # Intermediate logits via logit lens
                interm_logits = self.lm_head(self.norm(hidden_states[l][:, -1]))

                one_hot = interm_logits[:, next_token_id].sum()
                grad = torch.autograd.grad(
                    one_hot, [attentions[l]], retain_graph=True
                )[0]
                grad_last_query = torch.nan_to_num(
                    grad[:, :, -1, :],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                # Gradient w.r.t. prompt text tokens and generated context tokens.
                grad_prompt_text = grad_last_query[:, :, :prompt_length][
                    ..., prompt_text_mask
                ].clamp(min=0)
                grad_generated_text = grad_last_query[:, :, prompt_length:].clamp(min=0)
                if grad_generated_text.shape[-1] > 0:
                    grad_all_text = torch.cat(
                        [grad_prompt_text, grad_generated_text], dim=-1
                    )
                else:
                    grad_all_text = grad_prompt_text

                # Gradient w.r.t. image tokens
                grad_img = grad_last_query[:, :, :prompt_length][
                    ..., prompt_image_mask
                ].clamp(min=0)

                new_token_grads.append(grad_img.detach())
                heads_topk_norm_img.append(
                    torch.nan_to_num(topk_norm(grad_img, k=k_img), nan=0.0)
                )
                if grad_all_text.shape[-1] > 0:
                    heads_topk_norm_text.append(
                        torch.nan_to_num(
                            topk_norm(grad_all_text, k=k_all_text),
                            nan=0.0,
                        )
                    )
                else:
                    heads_topk_norm_text.append(torch.zeros_like(heads_topk_norm_img[-1]))

            # Aggregate across layers for this token
            all_heads_topk_norm_img.append(torch.cat(heads_topk_norm_img))
            all_heads_topk_norm_text.append(torch.cat(heads_topk_norm_text))
            new_token_grads = torch.cat(new_token_grads)  # [num_layers_used, heads, N]
            all_new_tokens_grads.append(new_token_grads)

            # Extend input sequence with the current token
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

        # ========== Aggregation ==========
        # [num_tokens, num_layers_used, heads, N]
        all_new_tokens_grads = torch.stack(all_new_tokens_grads)
        all_new_tokens_grads = torch.nan_to_num(
            all_new_tokens_grads, nan=0.0, posinf=0.0, neginf=0.0
        )
        all_new_tokens_grads = min_max(all_new_tokens_grads) # FIX
        # [num_tokens, num_layers_used * heads]
        all_heads_topk_norm_img = torch.nan_to_num(
            torch.stack(all_heads_topk_norm_img), nan=0.0, posinf=0.0, neginf=0.0
        )
        all_heads_topk_norm_text = torch.nan_to_num(
            torch.stack(all_heads_topk_norm_text), nan=0.0, posinf=0.0, neginf=0.0
        )

        # --- Head filtering (Eq. 5): w = (S_img - S_text)^+ ---
        filtering_weights = (all_heads_topk_norm_img - all_heads_topk_norm_text).clamp(min=0)
        # [num_tokens, num_layers_used, heads]
        num_layers_used = self.num_layers - starting_depth
        num_heads = all_new_tokens_grads.shape[2]
        filtering_weights = filtering_weights.view(
            num_tokens_to_generate, num_layers_used, num_heads
        )

        # --- Filtered per-token heatmaps (Eq. 5): weighted by head filtering ---
        filtered_grads = all_new_tokens_grads * filtering_weights.unsqueeze(-1)
        per_token_flat = filtered_grads.sum(dim=[1, 2])  # [num_tokens, N]
        per_token_heatmaps = rearrange(per_token_flat, "t (h w) -> t h w", h=h, w=w)
        for t in range(num_tokens_to_generate):
            hm = per_token_heatmaps[t]
            if hm.max() > hm.min():
                per_token_heatmaps[t] = min_max(hm)

        # --- Unfiltered per-token heatmaps: plain sum over all heads/layers ---
        unfiltered_flat = all_new_tokens_grads.sum(dim=[1, 2])  # [num_tokens, N]
        per_token_heatmaps_unfiltered = rearrange(unfiltered_flat, "t (h w) -> t h w", h=h, w=w)
        for t in range(num_tokens_to_generate):
            hm = per_token_heatmaps_unfiltered[t]
            if hm.max() > hm.min():
                per_token_heatmaps_unfiltered[t] = min_max(hm)

        # --- Token weights delta^t (Eq. 6) ---
        num_gen_tokens = all_heads_topk_norm_img.shape[0]
        delta_t = (
            all_heads_topk_norm_img.view(num_gen_tokens, -1).max(dim=-1)[0]
            - all_heads_topk_norm_text.view(num_gen_tokens, -1).max(dim=-1)[0]
        ).clamp(min=0)  # [num_tokens]

        # --- Filtered sentence-level heatmap (Eq. 6) ---
        sentence_flat = (
            filtered_grads * delta_t[:, None, None, None]
        ).sum(dim=[0, 1, 2])  # [N]
        sentence_heatmap = rearrange(sentence_flat, "(h w) -> h w", h=h, w=w)
        if sentence_heatmap.max() > sentence_heatmap.min():
            sentence_heatmap = min_max(sentence_heatmap)

        # --- Unfiltered sentence-level heatmap ---
        unfiltered_sentence_flat = all_new_tokens_grads.sum(dim=[0, 1, 2])  # [N]
        sentence_heatmap_unfiltered = rearrange(unfiltered_sentence_flat, "(h w) -> h w", h=h, w=w)
        if sentence_heatmap_unfiltered.max() > sentence_heatmap_unfiltered.min():
            sentence_heatmap_unfiltered = min_max(sentence_heatmap_unfiltered)

        return DexarResult(
            per_token_heatmaps=per_token_heatmaps.detach(),
            per_token_heatmaps_unfiltered=per_token_heatmaps_unfiltered.detach(),
            token_weights=delta_t.detach(),
            sentence_heatmap=sentence_heatmap.detach(),
            sentence_heatmap_unfiltered=sentence_heatmap_unfiltered.detach(),
            tokens=token_strings,
        )
