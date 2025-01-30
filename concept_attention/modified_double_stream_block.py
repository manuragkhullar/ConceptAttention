import torch
from torch import nn, Tensor
import einops
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

from flux.modules.layers import Modulation, SelfAttention
from flux.math import apply_rope


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = scaled_dot_product_attention(q, k, v)
    x = einops.rearrange(x, "B H L D -> B L (H D)")

    return x

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query, 
    key, 
    value,
    attn_mask=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value

class ModifiedDoubleStreamBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        # Store information needed for visualization
        self.concept_key_vectors = []
        self.image_key_vectors = []
        self.concept_query_vectors = []
        self.image_query_vectors = []
        self.concept_value_vectors = []
        self.image_value_vectors = []
        self.image_output_vectors = []
        self.concept_output_vectors = []
        self.concept_heatmaps = []
        self.cross_attention_maps = []
        self.text_value_vectors = []
        self.text_query_vectors = []
        self.text_key_vectors = []

    def clear_cached_vectors(self):
        """Clear out cached vectors"""
        self.concept_key_vectors = []
        self.image_key_vectors = []
        self.concept_query_vectors = []
        self.image_query_vectors = []
        self.concept_value_vectors = []
        self.image_value_vectors = []
        self.image_output_vectors = []
        self.text_query_vectors = []
        self.text_key_vectors = []
        self.text_value_vectors = []
        self.concept_output_vectors = []
        self.concept_heatmaps = []
        self.cross_attention_maps = []

    @torch.no_grad()
    def forward(
        self, 
        img: Tensor, 
        txt: Tensor, 
        vec: Tensor, 
        pe: Tensor, 
        concepts: Tensor, 
        concept_vec: Tensor,
        concept_pe: Tensor,
        null_txt: Tensor,
        null_txt_vec: Tensor,
        null_txt_pe: Tensor,
        joint_attention_kwargs=None,
        **kwargs
    ) -> tuple[Tensor, Tensor]:
        assert concept_vec is not None, "Concept vectors must be provided for this implementation."
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        concept_mod1, concept_mod2 = self.txt_mod(concept_vec)
        null_txt_mod1, null_txt_mod2 = self.txt_mod(null_txt_vec)
        # Prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = einops.rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        # Prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = einops.rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        # Prepare concepts for attention
        concept_modulated = self.txt_norm1(concepts)
        concept_modulated = (1 + concept_mod1.scale) * concept_modulated + concept_mod1.shift
        concept_qkv = self.txt_attn.qkv(concept_modulated)
        concept_q, concept_k, concept_v = einops.rearrange(concept_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        concept_q, concept_k = self.txt_attn.norm(concept_q, concept_k, concept_v)
        # Prepare null text for attention
        null_txt_modulated = self.txt_norm1(null_txt)
        null_txt_modulated = (1 + null_txt_mod1.scale) * null_txt_modulated + null_txt_mod1.shift
        null_txt_qkv = self.txt_attn.qkv(null_txt_modulated)
        null_txt_q, null_txt_k, null_txt_v = einops.rearrange(null_txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        null_txt_q, null_txt_k = self.txt_attn.norm(null_txt_q, null_txt_k, null_txt_v)
        # Save concept key vectors
        self.concept_key_vectors.append(concept_k.detach().cpu())
        self.concept_query_vectors.append(concept_q.detach().cpu())
        self.concept_value_vectors.append(concept_v.detach().cpu())
        self.image_key_vectors.append(img_k.detach().cpu())
        self.image_query_vectors.append(img_q.detach().cpu())
        self.image_value_vectors.append(img_v.detach().cpu())
        self.text_key_vectors.append(txt_k.detach().cpu())
        self.text_query_vectors.append(txt_q.detach().cpu())
        self.text_value_vectors.append(txt_v.detach().cpu())
        # Concatenate each modality
        # q = torch.cat((txt_q, img_q, concept_q, null_txt_q), dim=2)
        # k = torch.cat((txt_k, img_k, concept_k, null_txt_k), dim=2)
        # v = torch.cat((txt_v, img_v, concept_v, null_txt_v), dim=2)
        # # Concatenate concept pe
        # # pe = torch.cat((pe, concept_pe), dim=1)
        # # Apply the rotary position encoding
        # q, k = apply_rope(q, k, pe)
        # """
        #     Here we jointly do the concept attention and the image/text attention
        #     In order to ensure that the image modality influences the concepts, but the concept
        #     does not influence the text or the image we need to use an attention mask. 
        # """
        # # Make indices arrays for each modality 
        # txt_start, txt_end = 0, txt.shape[1]
        # img_start, img_end = txt_end, txt_end + img.shape[1]
        # concept_start, concept_end = img_end, img_end + concepts.shape[1]
        # null_txt_start, null_txt_end = concept_end, concept_end + null_txt.shape[1]
        # # Make the mask for the attention
        # attn_mask = torch.zeros(q.shape[2], k.shape[2], device=q.device, dtype=q.dtype)
        # # Set the image/text section to 1
        # attn_mask[
        #     txt_start:img_end, # Text and image tokens
        #     txt_start:img_end # Text and image tokens
        # ] = 1
        # # Set the concept/concept section to one (concept self-attention)
        # attn_mask[
        #     concept_start:concept_end, # Concept tokens
        #     concept_start:concept_end # Concept tokens
        # ] = 1 
        # # Set the section allowing concept to pull from image (NOT the other way around!!)
        # attn_mask[
        #     concept_start:concept_end,
        #     img_start:concept_end
        # ] = 1
        # # Allow self-attention for the null text
        # attn_mask[
        #     null_txt_start:null_txt_end,
        #     null_txt_start:null_txt_end
        # ] = 1
        # # Do cross atttention between null text and the image tokens
        # attn_mask[
        #     null_txt_start:null_txt_end,
        #     img_start:img_end
        # ] = 1
        # attn_mask[
        #     img_start:img_end,
        #     null_txt_start:null_txt_end
        # ]
        # # Do the attention operation
        # attn = F.scaled_dot_product_attention(
        #     q, 
        #     k, 
        #     v,
        #     attn_mask=attn_mask
        # )
        # # Make cross attention heatmap
        # img_txt_q = q[:, :, :txt.shape[1] + img.shape[1]]
        # img_txt_k = k[:, :, :txt.shape[1] + img.shape[1]]
        # attention_weights = einops.einsum(
        #     img_txt_q,
        #     img_txt_k,
        #     "batch heads queries dim, batch heads keys dim -> batch heads queries keys"
        # )
        # attention_weights = torch.softmax(attention_weights, dim=-1)
        # attention_weights = einops.reduce(
        #     attention_weights,
        #     "batch heads queries keys -> queries keys",
        #     reduction="mean"
        # )
        # cross_attention_weights = attention_weights[txt.shape[1]:, :txt.shape[1]]
        # cross_attention_weights = einops.rearrange(
        #     cross_attention_weights,
        #     "(h w) concepts -> concepts h w",
        #     h=64,
        #     w=64
        # )
        # self.cross_attention_maps.append(
        #     cross_attention_weights.detach().cpu()
        # )
        # # Separate the image and text attentions
        # txt_attn = attn[:, :, txt_start:txt_end]
        # img_attn = attn[:, :, img_start:img_end]
        # concept_attn = attn[:, :, concept_start:concept_end]
        # null_txt_attn = attn[:, :, null_txt_start:null_txt_end]
        ########## Do the text-image joint attention ##########
        text_image_q = torch.cat((txt_q, img_q), dim=2)
        text_image_k = torch.cat((txt_k, img_k), dim=2)
        text_image_v = torch.cat((txt_v, img_v), dim=2)
        # Apply rope
        text_image_q, text_image_k = apply_rope(text_image_q, text_image_k, pe)
        # Do the attention operation
        text_image_attn = F.scaled_dot_product_attention(
            text_image_q, 
            text_image_k,
            text_image_v
        )
        # Separate the text and image attentions
        txt_attn = text_image_attn[:, :, :txt.shape[1]]
        img_attn = text_image_attn[:, :, txt.shape[1]:]
        ########## Do the concept-image joint attention ##########
        concept_image_q = torch.cat((concept_q, img_q), dim=2)
        concept_image_k = torch.cat((concept_k, img_k), dim=2)
        concept_image_v = torch.cat((concept_v, img_v), dim=2)
        # Apply rope
        concept_image_q, concept_image_k = apply_rope(concept_image_q, concept_image_k, concept_pe)
        if joint_attention_kwargs is not None:
            concept_cross_attention = joint_attention_kwargs.get("concept_cross_attention", True)
            concept_self_attention = joint_attention_kwargs.get("concept_self_attention", True)
            if concept_cross_attention and not concept_self_attention:
                # Do cross attention only between concepts and image
                concept_only_q = concept_image_q[:, :, :concepts.shape[1]]
                image_only_k = concept_image_k[:, :, concepts.shape[1]:]
                # Do the attention operation
                concept_attn = scaled_dot_product_attention(
                    concept_only_q,
                    image_only_k,
                    img_v
                )
            elif concept_self_attention and not concept_cross_attention:
                concept_q = concept_image_q[:, :, :concepts.shape[1]]
                concept_k = concept_image_k[:, :, :concepts.shape[1]]
                # Do the attention operation
                concept_attn = scaled_dot_product_attention(
                    concept_q,
                    concept_k,
                    concept_v
                )
            elif concept_cross_attention and concept_self_attention: 
                # Do the attention operation
                concept_image_attn = F.scaled_dot_product_attention(
                    concept_image_q, 
                    concept_image_k, 
                    concept_image_v,
                )
                # Separate the concept and image attentions
                concept_attn = concept_image_attn[:, :, :concepts.shape[1]]
            else:
                # Neither self or cross. 
                concept_attn = concept_v
        else:
            # Do both cross and self attention
            concept_image_attn = F.scaled_dot_product_attention(
                concept_image_q, 
                concept_image_k, 
                concept_image_v,
            )
            # Separate the concept and image attentions
            concept_attn = concept_image_attn[:, :, :concepts.shape[1]]

        ########## Do the null text-image joint attention ##########
        null_text_image_q = torch.cat((null_txt_q, img_q), dim=2)
        null_text_image_k = torch.cat((null_txt_k, img_k), dim=2)
        null_text_image_v = torch.cat((null_txt_v, img_v), dim=2)
        # Apply rope
        null_text_image_q, null_text_image_k = apply_rope(null_text_image_q, null_text_image_k, null_txt_pe)
        # Do the attention operation
        null_text_image_attn = F.scaled_dot_product_attention(
            null_text_image_q, 
            null_text_image_k, 
            null_text_image_v
        )
        # Separate the null text and image attentions
        null_txt_attn = null_text_image_attn[:, :, :null_txt.shape[1]]
        img_null_txt_attn = null_text_image_attn[:, :, null_txt.shape[1]:]
        # Rearrange the attention tensors
        txt_attn = einops.rearrange(txt_attn, "B H L D -> B L (H D)")

        if joint_attention_kwargs is not None and joint_attention_kwargs.get("keep_head_dim", False):
            self.concept_output_vectors.append(
                concept_attn.detach().cpu()
            )
            self.image_output_vectors.append(
                img_attn.detach().cpu()
            )
            concept_attn = einops.rearrange(concept_attn, "B H L D -> B L (H D)")
            img_attn = einops.rearrange(img_attn, "B H L D -> B L (H D)")
        else:
            concept_attn = einops.rearrange(concept_attn, "B H L D -> B L (H D)")
            img_attn = einops.rearrange(img_attn, "B H L D -> B L (H D)")
            self.concept_output_vectors.append(
                concept_attn.detach().cpu()
            )
            self.image_output_vectors.append(
                img_attn.detach().cpu()
            )
        null_txt_attn = einops.rearrange(null_txt_attn, "B H L D -> B L (H D)")
        # Compute and save concept heatmap
        combined_q = torch.cat((concept_q, img_q), dim=2)
        combined_k = torch.cat((concept_k, img_k), dim=2)
        concept_heatmap = einops.einsum(
            combined_q, 
            combined_k,
            "B H L_1 D, B H L_2 D -> B H L_2 L_1"
        )
        concept_heatmap = concept_heatmap[:, :, :concepts.shape[1], concepts.shape[1]:]
        self.concept_heatmaps.append(
            concept_heatmap.detach().cpu()
        )
        # Do the block updates
        # Calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # Can I do the decomposition here? Using a basis formed by (img_mod1.gate * self.img_attn.proj(concepts))
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        # Calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        # Calculate the concept blocks
        concepts = concepts + concept_mod1.gate * self.txt_attn.proj(concept_attn)
        concepts = concepts + concept_mod2.gate * self.txt_mlp((1 + concept_mod2.scale) * self.txt_norm2(concepts) + concept_mod2.shift) 
        # Calculate the null block updates
        null_txt = null_txt + null_txt_mod1.gate * self.txt_attn.proj(null_txt_attn)
        null_txt = null_txt + null_txt_mod2.gate * self.txt_mlp((1 + null_txt_mod2.scale) * self.txt_norm2(null_txt) + null_txt_mod2.shift)

        return img, txt, concepts, null_txt