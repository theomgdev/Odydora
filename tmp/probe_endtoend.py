"""The number that actually decides the design: end-to-end training tok/s of
a real 1.7B-geometry model, with the core at k call sites vs no core at all.
The core's launch overhead only matters RELATIVE to the transformer's own cost."""
import time, torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from odydora.adapter import attach_core, detach_core, count_parameters

dev = "cuda"
torch.manual_seed(0)

# Qwen3-1.7B真 geometry: 28 layers, hidden 2048, 16 heads, 8 kv heads
cfg = Qwen3Config(num_hidden_layers=28, hidden_size=2048, intermediate_size=6144,
                  num_attention_heads=16, num_key_value_heads=8,
                  vocab_size=151936, max_position_embeddings=4096,
                  head_dim=128)
print("building Qwen3-1.7B geometry in fp16 (random weights, no download)...")
model = Qwen3ForCausalLM(cfg).half().to(dev)
total = sum(p.numel() for p in model.parameters())
print(f"  {total/1e9:.2f}B params, {torch.cuda.memory_allocated()/2**30:.2f} GiB weights\n")

def bench_fwdbwd(m, T=512, reps=3, warm=1):
    ids = torch.randint(0, 1000, (1, T), device=dev)
    def once():
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        m.zero_grad(set_to_none=True)
    for _ in range(warm): once()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): once()
    torch.cuda.synchronize()
    return T * reps / (time.time() - t0)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()   # what prepare_model_for_kbit_training does
model.requires_grad_(False)
model.model.layers[0].mlp.down_proj.weight.requires_grad_(True)
base = bench_fwdbwd(model)
print(f"baseline, no core:            {base:8,.0f} tok/s  (fwd+bwd, T=512, grad ckpt on)")

print("
stride sweep, core at ALL 28 layers (the full 'frontal cortex'):")
for stride in (16, 64, 128, 256):
    m = Qwen3ForCausalLM(cfg).half().to(dev)
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    m = attach_core(m, n=64, layers=None, stride=stride)
    m.odydora_reset_state()
    try:
        tps = bench_fwdbwd(m)
        _, core = count_parameters(m)
        print(f"  stride {stride:4d}, 28 sites: {tps:8,.0f} tok/s "
              f"({100*tps/base:5.1f}% of baseline)  core {core:,} params")
    except RuntimeError as e:
        print(f"  stride {stride:4d}: FAILED {str(e)[:70]}")
    del m; torch.cuda.empty_cache()

print("
call-site sweep at stride=64:")
for k, layers in ((1,(27,)), (4,(6,13,20,27)), (28,None)):
    m = Qwen3ForCausalLM(cfg).half().to(dev)
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    m = attach_core(m, n=64, layers=layers, stride=64)
    sites = len(m.odydora_layers)
    m.odydora_reset_state()
    try:
        tps = bench_fwdbwd(m)
        _, core = count_parameters(m)
        print(f"core at {sites:2d} site(s): {tps:8,.0f} tok/s  "
              f"({100*tps/base:5.1f}% of baseline)  core params {core:,}")
    except RuntimeError as e:
        print(f"core at {sites:2d} site(s): FAILED {str(e)[:80]}")
    del m; torch.cuda.empty_cache()
