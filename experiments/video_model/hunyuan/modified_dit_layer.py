from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from diffusers.loaders import FromOriginalModelMixin

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor

from diffusers.models.normalization import AdaLayerNormZero
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ModifiedHunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_with_concept: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        #################### Do ConceptAttention operation  ####################
        # 3. Concept embedding normalization
        norm_concept_hidden_states, concept_gate_msa, concept_shift_mlp, concept_scale_mlp, concept_gate_mlp = self.norm1_context(
            concept_hidden_states, 
            emb=temb
        )
        # 4. Do the concept attention operation
        _, concept_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_concept_hidden_states,
            attention_mask=attention_mask_with_concept,
            image_rotary_emb=freqs_cis,
        )
        ########################################################################
        # 5. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 6. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        #################### Create concept attention maps  ####################
        # 7. Now do the concept attention operation to create the maps
        concept_attention_maps = einops.einsum(
            attn_output,
            concept_attn_output,
            "batch num_patch dim, batch concepts dim -> batch concepts num_patch"
        )
        concept_attention_dict = {
            "concept_attention_maps": concept_attention_maps.cpu()
        }
        # 8. Modulation and residual connection
        concept_hidden_states = concept_hidden_states + concept_attn_output * concept_gate_msa.unsqueeze(1)
        norm_concept_hidden_states = self.norm2_context(concept_hidden_states)
        norm_concept_hidden_states = norm_concept_hidden_states * (1 + concept_scale_mlp[:, None]) + concept_shift_mlp[:, None]
        # 9. Feed-forward
        concept_ff_output = self.ff_context(norm_concept_hidden_states)
        concept_hidden_states = concept_hidden_states + concept_gate_mlp.unsqueeze(1) * concept_ff_output
        ########################################################################

        return hidden_states, encoder_hidden_states, concept_hidden_states, concept_attention_dict