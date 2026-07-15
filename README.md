# Optimizing Qwen3-0.6B with LLM Compressor 

This code quantizes [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) from full-precision (BF16) down to 4-bit weights using **GPTQ (W4A16)** via the [`llm-compressor`](https://github.com/vllm-project/llm-compressor) library and verifies the compressed model still produces reasonable output.

This is a post-training quantization (PTQ) and it requires no retraining, no fine-tuning, just a single calibrated pass over the model's weights.

## What is quantization?

Quantization is the process of representing a model's weights (and sometimes its activations) with fewer bits than the precision they were trained in. A model trained in 16-bit floating point (BF16) stores every weight as a 16-bit number, meaning that each pararmeter consumes 2 bytes of GPU memory, quantizing to 4 bits maps those values onto a much smaller set of levels, so each weight takes a quarter of the space.

Think of it as compression for neural networks instead of storing every weight at full fidelity, you store a lower-precision approximation plus a small amount of metadata (scales and zero-points) that lets you reconstruct values close to the originals. Done naively this might loses accuracy, but calibrated methods like **GPTQ** choose the quantized values to minimize the error that actually matters for the model's output, so quality holds up better than plain rounding.

Precision is usually described with a `WxAy` shorthand, `x`-bit **W**eights and `y`-bit **A**ctivations. This code specifically uses **W4A16** 4-bit weights, 16-bit activations.

## Why quantization matters for inference

A large language model is expensive to serve mostly because of its size. Quantization attacks that cost directly:

- **Less memory**: 4-bit weights use ~4x less VRAM/RAM than BF16, so a model that wouldn't fit on a given GPU suddenly does, or you can fit a bigger batch, a longer context, or more concurrent requests on the same hardware.
- **Faster inference**: LLM inference is largely **memory-bandwidth bound** the bottleneck is moving weights from memory to the compute units, not the arithmetic itself. Smaller weights mean less data to move per token, which translates directly into higher throughput and lower latency.
- **Cheaper deployment**: fewer/smaller GPUs to serve the same workload, smaller downloads, and lower operational cost at scale.
- **Minimal quality loss**: with calibrated PTQ, these gains come at a small, *measurable* accuracy cost, which is exactly what this script verifies with generation samples and perplexity numbers rather than asking you to take it on faith.

In short: quantization is one of the highest-leverage optimizations for taking a model from "trained" to "cheaply and quickly served in production."

## Before you start quantizing, evaluate first

Quantization is never free, it always trades some accuracy for size and speed, and how much you lose is specific to *your* model, *your* prompts, and *your* quality bar. So don't reach for a heavyweight technique before you know you need one. A sensible progression looks like this:

1. **Start at the serving layer.** Before adopting any offline quantization pipeline, try the quantization your inference server already offers. For example, vLLM can apply on-the-fly schemes (like FP8 or other runtime quantization) with a single flag, no calibration pass, no separate artifact to manage. It's the cheapest thing to try. You can something like `vllm serve meta-llama/Llama-3-70b-instruct --quantization fp8` but you can't go down to 4 bits only with it.
2. **Measure it against your use case.** Run your real workload through the quantized model and evaluate the outputs you actually care about, task accuracy, perplexity, latency, throughput, not a generic benchmark. If quality holds up, you're done; ship it. You must use a EDD (Evaluation Developement Driven) approach where you have an eval in place to determine any model adjustment.
3. **Escalate only if it doesn't hold up.** If the serving-layer approach isn't accurate or precise enough, *then* move to calibrated, offline techniques that recover more quality: GPTQ, AWQ, QAT, SmoothQuant, and friends. Each is more work than the last, so only pay that cost once the simpler option has demonstrably failed your evaluation.

**This post assumes you've already done that evaluation and landed on GPTQ (W4A16) as the technique that fits your accuracy and footprint targets.** Everything below is the *how*, it takes GPTQ as a given rather than re-arguing the choice. If you haven't run that comparison yet, do it first; you may find the serving layer is all you need.

## How this use case landed on GPTQ (W4A16)

To make the reasoning concrete, here's the exact path that led to GPTQ for this use case:

1. **Serving-layer quantization showed an accuracy decrease.** Running the workload through vLLM's on-the-fly quantization was the first thing I tried using `--quantization fp8`. It was cheap to turn on, but the evaluation showed the quality drop was larger than acceptable for the use case, so the runtime option alone didn't clear the bar. 
2. **The footprint target required going to 4-bit weights.** The goal was to push the model down to 4-bit (roughly a 4× reduction), but *naive* 4-bit rounding at 4-bit only leaves **16 representable levels per weight**, so rounding each weight to the nearest level throws away too much information and the output degrades noticeably. There simply isn't enough precision left to be careless with it.
3. **GPTQ is what makes 4-bit viable.** Instead of rounding blindly, GPTQ uses a small calibration set to measure how each weight actually affects the model's output, then places the 16 available levels where they minimize that output error. That's the difference between "4-bit that's unusable" and "4-bit that holds up on the eval", and it's precisely why the escalation from the serving layer lands on GPTQ rather than plain 4-bit rounding.

### Why "weights" and "activations" are quantized separately

The `WxAy` scheme splits into two very different things, and understanding why is what justifies the `W4A16` choice:

- **Weights (the `W`)** are *static*. They're fixed once the model is trained and identical for every request, so you can analyze them offline, calibrate carefully, and store them at low precision once. They're also the bulk of what makes a model big and what has to be streamed from memory on every token, **so quantizing weights is where almost all the memory and bandwidth savings come from**. This is the safe, high-payoff target, and it's why we go aggressive here: **4-bit**.
- **Activations (the `A`)** are *dynamic*. They're computed at runtime from the input and change with every prompt, so they can't be pre-calibrated the same way, and they tend to contain large outliers that low precision handles badly. Quantizing activations mainly speeds up the arithmetic (letting the GPU do lower-precision math), but it's riskier for quality and often needs extra machinery (e.g. SmoothQuant) to tame those outliers.

`W4A16` is the deliberate consequence of that asymmetry: **squeeze the weights hard (4-bit) because that's where the savings are and it's safe to calibrate, but leave activations at full 16-bit** because touching them buys less and costs more quality. LLM decoding is mostly *memory-bandwidth bound*, the bottleneck is moving weights, not doing math, so shrinking weights while keeping activations at 16-bit captures the large majority of the benefit at the smallest quality risk. Schemes like `W4A8` or `W8A8` only make sense once activation math genuinely becomes your bottleneck and you're willing to add the extra calibration to protect quality.

## The llm compressor

For this use case, I'm using llm-compressor which is a vLLM project. 

The core API is `oneshot` which is quite straightforward, you give it a model, a small calibration dataset, and a "recipe" describing how to quantize, and it produces a smaller model ready to be served directly by vLLM.

```python
oneshot(
    model="model-name",           
    dataset="dataset-name",       
    recipe=recipe,                
    output_dir="./output",        
    num_calibration_samples=256,  
    max_seq_length=4096,          
)
```

The name "oneshot" reflects that this happens in a **single pass** over the calibration data, the model's original weights are never retrained, only re-represented at lower precision.

## Why a calibration dataset?

GPTQ doesn't just round every weight down to 4 bits naively. It uses a small sample of real text (here, ultrachat_200k) to measure how each weight actually affects the model's output, then chooses quantized values that minimize that error. This is what makes GPTQ meaningfully more accurate than plain rounding, and why only a few hundred samples are needed; calibration is fast relative to training.

### Quantization schemes 

| Scheme | Weights | Activations | Quantized-layer reduction | Quality impact |
|---|---|---|---|---|
| W8A16 | 8-bit | 16-bit | ~50% | Minimal |
| **W4A16** | **4-bit** | **16-bit** | **~75%** | **Low–Moderate** |
| W8A8 | 8-bit | 8-bit | ~50% | Low |
| W4A8 | 4-bit | 8-bit | ~75% | Moderate |

## What the code does, step by step

1. **Defines the recipe** and prints it for visibility.
2. **Runs `oneshot`** to quantize `Qwen/Qwen3-0.6B` using ultrachat_200k as calibration data, saving the result to `models/Qwen3-0.6B-W4A16`. If that folder already exists, this step is skipped, so re-running the script won't redo the (slower) quantization pass.
3. **Compares folder size** between the original and quantized model directories and prints the percentage reduction.
4. **Generates text** from both the base and quantized model on the same prompt, so you can eyeball whether output quality held up.
5. **Computes perplexity** a standard measure of how well a language model predicts held-out text, lower is better, for both models on the ultrachat_200k test split, and reports the difference. This is the quantitative check that quantization didn't meaningfully hurt quality.

## Why the size reduction isn't a clean 75%

Going from 16-bit to 4-bit weights is a 4x (75%) reduction *in theory*, but only the `Linear` layer weights actually get quantized, the `lm_head`, embeddings, and normalization layers stay at full precision. For a small model like this one, those unquantized pieces make up a larger share of total parameters, so expect **roughly 40–50% total size reduction**, not 75%. This ratio improves with model size: a 70B model quantized the same way lands much closer to the theoretical 4x, since linear layers dominate an even larger fraction of the total.

## Requirements

```bash
pip install -r requirements.txt
```

You'll also need the base model downloaded locally to `models/Qwen3-0.6B` (or adjust `MODEL_DIR`).

## Usage

```bash
python main.py
```

Expected output includes: the recipe definition, a size comparison table, sample generations from both models, and a final perplexity comparison table.

## Understanding the output

A full run is captured in [`output.txt`](output.txt). Here's how to read it, section by section, and why a few numbers look the way they do.

### Size reduction: 42%

```
Original (BF16):    1.41 GB
Quantized (W4A16):  835.3 MB
Reduction:          42%
```

This is lower than the "~75%" the scheme suggests, and that's expected for a model this small (see [Why the size reduction isn't a clean 75%](#why-the-size-reduction-isnt-a-clean-75) above). The concrete reason here is bcause Qwen3-0.6B sets `tie_word_embeddings: true`, so the `lm_head` is the *same* tensor as the input embedding, and the recipe ignores `lm_head`. That shared embedding table is `vocab_size × hidden = 151936 × 1024 × 2 bytes ≈ 311 MB` of BF16 that never gets quantized. On a 0.6B model that fixed embedding cost dominates; on a 7B+ model the same recipe lands much closer to 4×. So 42% is correct, not a bug.

### Sample generations

```
Base Model
Prompt: What is AI inference?
Response: AI inference refers to the process of using artificial intelligence (AI) systems to make
decisions, predictions, or provide solutions based on data. It involves training a model with a
large dataset, then using that model to analyze new data... For example, in healthcare, AI
inference might

Quantized Model
Prompt: What is AI inference?
Response: AI inference refers to the process of using artificial intelligence (AI) to make decisions
or predictions based on data. It involves training models to understand patterns and making
predictions from that data. For example, in healthcare, AI inference might help identify diseases
by analyzing medical images or in finance, it might predict stock
```

Both models now produce a direct, coherent answer, and the two are nearly interchangeable the quantized response tracks the base almost sentence for sentence, only diverging in minor word choice. This is what a healthy W4A16 quantization is supposed to look like at the eyeball level.

The reason the samples are clean is that the prompt is fed through the chat template with `add_generation_prompt=True`, so the instruction-tuned model receives the exact format it was tuned and calibrated on. (Earlier runs that fed a raw completion string like `"AI Inference is"` made the model drift into a fill-in-the-blank quiz format — that was a prompting mismatch, not quantization damage.)

Treat a single greedy generation as a smoke test, not a quality metric — the perplexity number below is the trustworthy signal. Here it agrees with the eyeball read: both look good, and the measured gap is small.

### Perplexity: +5.9%

```
Base (BF16):       11.71
Quantized (W4A16): 12.40
Difference:        +0.69 (+5.9%)
```

This is the quantitative check, and it's the reassuring one: a small, single-digit-percent perplexity increase is exactly what a healthy W4A16 quantization should cost. Perplexity measures how well the model predicts held-out text (lower is better), so a +5.9% rise means the 4-bit weights preserved nearly all of the model's predictive quality. In other words, despite the odd-looking sample generation, the measured degradation is mild and expected.

