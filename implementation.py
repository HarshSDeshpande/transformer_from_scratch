# %%
import einops
from fancy_einsum import einsum
from dataclasses import dataclass
from easy_transformer import EasyTransformer
import torch
import torch.nn as nn
import numpy as np
import math
from easy_transformer.utils import get_corner, gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm

MAIN = __name__ == "__main__"

# %%
if MAIN:
    reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
#%%
if MAIN:
    tokens = reference_gpt2.to_tokens("Never forget who you are. The rest of the world will not. Wear it like armour, and it can never be used to hurt you.")
    tokens = tokens.cuda()
    logits,cache = reference_gpt2.run_with_cache(tokens)
    
# %%
@dataclass
class Config:
    d_model:int = 768
    debug:bool = True
    layer_norm_eps:float = 1e-5
    d_vocab:int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()
print(cfg)

# %%
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    random_input = torch.randn(shape).cuda()
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    random_input = torch.randint(100, 1000, shape).cuda()
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output

def load_gpt2_test(cls, gpt2_layer, input_name, cache_dict=cache.cache_dict):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    # Allow inputs of strings or tensors
    if isinstance(input_name, str): 
        reference_input = cache_dict[input_name]
    else:
        reference_input = input_name
    print("Input shape:", reference_input.shape)
    output = layer(reference_input)
    print("Output shape:", output.shape)
    reference_output = gpt2_layer(reference_input)
    print("Reference output shape:", reference_output.shape)

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct")
    return output

# %%
class LayerNorm(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self,residual):
        if self.cfg.debug:
            print("Residual:", residual.shape)
        residual = residual - einops.reduce(residual,"batch position d_model -> batch position 1","mean")
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1","mean") + self.cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized
# %%
if MAIN:
    _ = rand_float_test(LayerNorm, [2,4,768])
    _ = load_gpt2_test(LayerNorm, reference_gpt2.ln_final,"blocks.11.hook_resid_post")
# %%
class Embed(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=cfg.init_range)

    def forward(self, tokens):
        if self.cfg.debug:
            print("Tokens:", tokens.shape)
        embeddings = self.W_E[tokens,:]
        if self.cfg.debug:
            print("Embeddings:", embeddings.shape)
        return embeddings
# %%
if MAIN:
    rand_int_test(Embed, [2,4])
    load_gpt2_test(Embed,reference_gpt2.embed, tokens)
# %%
