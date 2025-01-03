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
    reference_text = "Never forget who you are. The rest of the world will not. Wear it like armour, and it can never be used to hurt you."
    tokens = reference_gpt2.to_tokens(reference_text)
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
class PosEmbed(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx,cfg.d_model))
        nn.init.normal_(self.W_pos,std=self.cfg.init_range)

    def forward(self,tokens):
        if self.cfg.debug:
            print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1),:]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug:
            print("Positional embedding:", pos_embed.shape)
        return pos_embed
# %%
if MAIN:
    rand_int_test(PosEmbed,[2,4])
    load_gpt2_test(PosEmbed, reference_gpt2.pos_embed,tokens)
# %%
if MAIN:
    import pysvelte
    pysvelte.AttentionMulti(tokens=reference_gpt2.to_str_tokens(reference_text),attention = cache['blocks.0.attn.hook_attn'][0].permute(1,2,0)).show()
# %%
class Attention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads,cfg.d_model,cfg.d_head)))
        nn.init.normal_(self.W_Q,std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads,cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads,cfg.d_model,cfg.d_head)))
        nn.init.normal_(self.W_K,std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads,cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads,cfg.d_model,cfg.d_head)))
        nn.init.normal_(self.W_V,std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads,cfg.d_head)))

        self.W_O = nn.Parameter(torch.empty((cfg.n_heads,cfg.d_head,cfg.d_model)))
        nn.init.normal_(self.W_O,std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

        self.register_buffer("IGNORE",torch.tensor(-1e5,dtype=torch.float32,device="cuda"))
    
    def forward(self,normalized_resid_pre):
        if self.cfg.debug:
            print("Normalized residual pre:", normalized_resid_pre.shape)
        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        
        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1)

        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z, self.W_O) + self.b_O
        return attn_out

    def apply_causal_mask(self,attn_scores):
        mask = torch.triu(torch.ones(attn_scores.size(-2),attn_scores.size(-1),device=attn_scores.device),diagonal=1).bool()
        attn_scores.masked_fill_(mask,self.IGNORE)
        return attn_scores
# %%
if MAIN:
    rand_float_test(Attention,[2,4,768])
    load_gpt2_test(Attention,reference_gpt2.blocks[0].attn,cache["blocks.0.ln1.hook_normalized"])
# %%
class MLP(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model,cfg.d_mlp)))
        nn.init.normal_(self.W_in,std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp,cfg.d_model)))
        nn.init.normal_(self.W_out,std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, normalized_resid_mid):
        if self.cfg.debug:
            print("Normalized_resid+mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out
# %%
if MAIN:
    rand_float_test(MLP,[2,4,768])
    load_gpt2_test(MLP,reference_gpt2.blocks[0].mlp,cache["blocks.0.ln2.hook_normalized"])
# %%
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg) 

    def forward(self,resid_pre):
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)
        resid_mid = resid_pre + attn_out
        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post
# %%
if MAIN:
    rand_float_test(TransformerBlock, [2, 4, 768])
    load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])
# %%
class Unembed(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model,cfg.d_vocab)))
        nn.init.normal_(self.W_U,std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab),requires_grad= False))

    def forward(self, normalized_resid_final):
        if self.cfg.debug:
            print("Normalized_resid_final:", normalized_resid_final.shape)
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
        return logits
# %%
if MAIN:
    rand_float_test(Unembed, [2, 4, 768])
    load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])
# %%
class DemoTransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self,tokens):
        embeddings = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        resid_pre = embeddings + pos_embed
        for block in self.blocks:
            resid_pre = block(resid_pre)
        normalized_resid_final = self.ln_final(resid_pre)
        logits = self.unembed(normalized_resid_final)
        return logits
# %%
if MAIN:
    rand_int_test(DemoTransformer, [2, 4])
    load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
# %%
if MAIN:
    demo_gpt2 = DemoTransformer(Config(debug=False))
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    demo_gpt2.cuda()

# %%
if MAIN:
    test_string = "I have a tender spot in my heart for cripples, bastards, and broken things."
    test_tokens = reference_gpt2.to_tokens(test_string).cuda()
    demo_logits = demo_gpt2(test_tokens)
# %%
def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:,:-1].gather(dim=-1,index=tokens[:,1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()
# %%
if MAIN:
    loss = lm_cross_entropy_loss(demo_logits, test_tokens)
    print(loss)
    print("Loss as average prob", (-loss).exp())
    print("Loss as 'uniform over this many variables'", (loss).exp())
    print("Uniform loss over the vocab",math.log(demo_gpt2.cfg.d_vocab))
# %%
if MAIN:
    test_string = "A mind needs books like a sword needs a whetstone."
    for i in tqdm.tqdm(range(100)):
        test_tokens = reference_gpt2.to_tokens(test_string).cuda()
        demo_logits = demo_gpt2(test_tokens)
        test_string += reference_gpt2.tokenizer.decode(demo_logits[-1,-1].argmax())
    print(test_string)
# %%
import datasets
import transformers
import plotly.express as px
# %%
batch_size = 8
num_epochs = 1
max_steps = 1000
log_every = 10
lr = 1e-3
weight_decay = 1e-2
model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)
# %%
if MAIN:
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
    print(dataset)
    print(dataset[0]['text'][:100])
    tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
    data_loader = torch.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# %%
if MAIN:
    model = DemoTransformer(model_cfg).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr,weight_decay=weight_decay)
# %%
if MAIN:
    losses = []
    print("Number of batches:",len(data_loader))
    for epoch in range(num_epochs):
        for c, batch in tqdm.tqdm(enumerate(data_loader)):
            tokens = batch['tokens'].cuda()
            logits = model(tokens)
            loss = lm_cross_entropy_loss(logits, tokens)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if c % log_every == 0:
                print(f"Step: {c}, Loss: {loss.item():.4f}")
            if c > max_steps:
                break
# %%
px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")