# CUDA Out of Memory — Root Cause & Fix

## GPU Budget

| Item | Memory |
|------|--------|
| GPU total | 5.64 GiB |
| SAM-3 video model weights (on GPU) | ~3.55 GiB |
| Free for everything else | ~2.06 GiB |

---

## Why It Crashed — Two Layers Deep

The OOM happens during `init_state()` when processing a video. There were **two layers** to the problem.

### Layer 1 — Frame loading in `io_utils.py` (already fixed)

`load_video_frames_from_video_file_using_cv2()` originally read every frame of the video, stacked them into a `(T, H, W, C)` NumPy array, converted it to a PyTorch tensor, and then did:

```python
video_tensor = video_tensor.cuda()  # ALL frames moved to GPU at once
```

For `bedroom.mp4` (200 frames at 1008×1008), this tensor alone exceeds the free VRAM. The `offload_video_to_cpu=True` flag was added to `init_state()` and wired through `run_sam3_inference.py` → `handle_request()` → `start_session()` → `model.init_state()` → `io_utils` to keep frames on CPU. ✅

### Layer 2 — `BatchedDatapoint` construction (the real crash for `bedroom.mp4`)

Even with frames correctly on CPU from the loader, `_construct_initial_input_batch()` in `sam3_video_inference.py` immediately undoes the offload:

```python
input_batch = BatchedDatapoint(img_batch=images, ...)  # images = CPU tensors ✓

# ← This recursively walks the dataclass and moves EVERYTHING to GPU,
#   including all 200 frames inside img_batch. Completely defeats the offload.
input_batch = copy_data_to_device(input_batch, device, non_blocking=True)  # 💥 OOM
```

`copy_data_to_device` is a recursive utility that calls `.to(device)` on every tensor it finds inside the dataclass hierarchy — including all video frames. The 2.27 GiB allocation reported in the crash is this exact moment.

---

## The Fix

`_construct_initial_input_batch` now accepts `offload_video_to_cpu` (passed from `init_state`). When `True`, only the non-image parts of the batch (`find_inputs`, etc.) are moved to GPU; `img_batch` stays on CPU.

### Files Changed

| File | Change |
|------|--------|
| `run_sam3_inference.py` | Passes `offload_video_to_cpu=True` in the `start_session` request |
| `sam3/model/sam3_video_predictor.py` | Reads the flag from the request and threads it into `start_session` → `model.init_state()` |
| `sam3/model/sam3_video_inference.py` | `init_state()` passes flag to `_construct_initial_input_batch`; the method conditionally skips `copy_data_to_device` on `img_batch` |

### Key code change (`sam3_video_inference.py`)

```python
def _construct_initial_input_batch(self, inference_state, images, offload_video_to_cpu=False):
    ...
    if offload_video_to_cpu:
        # Move only the lightweight index/prompt tensors to GPU.
        # img_batch stays on CPU — frames are moved to GPU one-by-one during inference.
        stages_on_device = copy_data_to_device(stages, device, non_blocking=True)
        input_batch = BatchedDatapoint(
            img_batch=images,          # CPU — never all on GPU at once
            find_text_batch=find_text_batch,
            find_inputs=stages_on_device,
            find_targets=[None] * num_frames,
            find_metadatas=[None] * num_frames,
        )
    else:
        input_batch = BatchedDatapoint(...)
        input_batch = copy_data_to_device(input_batch, device, non_blocking=True)
```

---

## Layer 3 — BFloat16 Precision & The Float32 Leak

While CPU offloading stopped the initial OOM, the 3.55 GiB model was still too large for a 5.64 GiB GPU during the actual attention calculations. We explicitly cast the model to `bfloat16` during setup (`model.to(dtype=torch.bfloat16)`), halving its memory footprint to ~1.77 GiB.

However, this triggered a persistent PyTorch crash: `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`. 

Tracing the model identified a **Float32 leak**: the `TransformerDecoderLayer` explicitly disables PyTorch's mixed precision (`autocast(enabled=False)`) for its FeedForward Network to prevent FP16 overflow. But because several upstream tracking masks and positional grids were initialized using default PyTorch float types (e.g. `torch.zeros()`, `torch.arange()`), they silently coerced the BFloat16 sequence tensors back into 32-bit floats during spatial additions. When these 32-bit tensors hit the explicitly disabled FFN, they crashed against the BFloat16 linear weights.

**The Fixes**:
1. **Positional Encodings**: Patched `PositionEmbeddingSine.forward()` and `gen_sineembed_for_position()` to dynamically inherit their output precision from the input context.
2. **Relative Position Bias**: Updated `_get_rpb_matrix()` in `decoder.py` to cast its spatial grids back to `reference_boxes.dtype` before entering the cross-attention blocks.
3. **Prompt Instantiations**: Forced `empty_geometric_prompt` in the video tracker and `visual_prompt_embed` in the image encoder to accept native `model_dtype` assignments.
4. **Valid Ratios & Text Features**: Ensured `valid_ratios` masks and language backbone outputs strictly align with the tracking coordinate sequences.
5. **Autocast Bypass**: Removed the `autocast(enabled=False)` bypass entirely from the FFN block. Since BFloat16 features the exact same exponent width as Float32, it natively solves the overflow risks associated with FP16 without requiring explicit 32-bit typecasting.

---

## Trade-off

CPU offloading means each frame is copied CPU→GPU on demand during inference, which adds minor PCIe transfer overhead. For a 5.64 GiB GPU with a 3.5 GiB model, this is the only viable path without hardware upgrades.

---

## Run It

To run inference on constrained GPUs (like a 6GB VRAM model), use the newly added memory-management flags:

```bash
conda activate sam3
python run_sam3_inference.py --type video --path inputs/videos/bedroom.mp4 --prompt "boy" --bfloat16 --offload_video
```

Output mask will be saved to `outputs/videos/bedroom_boy_mask.pt` (automatically appending the prompt).

### Reverting to Original High-Memory Mode
If you upgrade your hardware and no longer need these optimizations, simply drop the flags! The script defaults to standard PyTorch precision and loads video directly into GPU memory for maximum speed:

```bash
python run_sam3_inference.py --type video --path inputs/videos/bedroom.mp4 --prompt "boy"
```
