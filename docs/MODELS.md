# basal models on basal-lite

State on 2026-10-27/28: basal 2.4.1, vLLM 0.32.0, one RTX 3090 (24 GB).

`tools/check_models.py` takes the small files of every `Remek/basal-*` repository vLLM can load (config, tokenizer,
chat template, `basal.json`, `CALIBRATION*`) and checks the prompt, architecture, quantization, calibration and media
support. `-GGUF` and `-MLX` repositories are skipped: vLLM does not load them and basal has its own engines for them.

Legend: ✓ checked live against `basal-serve --mode vllm`, ● prompt identical offline and architecture supported by
vLLM, ○ does not fit on a 24 GB card.

## Summary

All 28 models give exactly basal's prompt tokens through basal-lite, and every option code is a single token. vLLM 0.32
supports every architecture. Gemma 4 models need the server chat template from `template.py` (two fixes, notes 4 and 5).

| model | architecture | weights | prompts | 24 GB | media in lite | status |
|---|---|---|---|---|---|---|
| basal-2.0-mini | Llama | 3.2 GB | 69/69 | ✓ | no encoder | ✓ vectors, example, throughput |
| basal-2.0-4.5B | Llama | 9.5 GB¹ | 69/69 | ✓ | no encoder | ✓ 53/53, 51/53, 52/53; think mode 53/53 (auto), 50/52 (force) |
| basal-2.0-max | Llama | 22.3 GB | 69/69 | ○ | no encoder | ● |
| basal-2.0-max-FP8 | Llama 11B, FP8 | 12.3 GB | 69/69 | ✓ | no encoder | ✓ 52/53, 52/53, 52/53 |
| basal-2.0-4.5B-FP8 | Llama, FP8 | 5.2 GB | 69/69 | ✓ | no encoder | ✓ 53/53, 51/53, 53/53 |
| basal-2.0-mini-FP8 | Llama, FP8 | 1.9 GB | 69/69 | ✓ | no encoder | ✓ 53/53, 53/53, 53/53 |
| basal-omni-pico | Qwen3.5 (Gated DeltaNet) | 1.5 GB | 69/69 | ✓ | no encoder | ✓ 53/53, 53/53, 53/53³ |
| basal-omni-1.0-mini | Gemma 4 E4B | 15.9 GB | 69/69 | ✓ | images, audio | ✓ text 52/53, 51/53, 53/53, media 16/16 |
| basal-omni-1.0-mini-FP8 | Gemma 4 E4B, FP8 | 12.3 GB | 69/69 | ✓ | images, audio | ✓ text 53/53, 51/53, media 16/16 |
| basal-omni-1.0-max | Gemma 4 Unified 12B | 23.9 GB | 69/69⁴ | ○ | refused⁵ | ● |
| basal-omni-1.0-max-FP8 | Gemma 4 Unified 12B, FP8 | 13.9 GB | 69/69⁴ | ✓ | refused⁵ (as basal) | ✓ 52/53, 52/53 |
| basal-omni-1.1-max | Gemma 4 Unified 12B | 23.9 GB | 69/69⁴ | ○ | refused⁵ | ● |
| basal-omni-1.0-ultra | Gemma 4 31B | 62.5 GB | 69/69⁴ | ○ | images | ● |
| basal-omni-1.0-ultra-FP8 | Gemma 4 31B, FP8 | 35.2 GB | 69/69⁴ | ○ | images | ● |
| basal-1.5-{mini,4.5B,max} | Llama, format 1 | 3.2 / 9.5 / 22.3 GB | 64/64⁶ | ● (max ○) | no encoder | ✓ 4.5B: 53/53, 53/53; no evidence⁷ |
| basal-1.5-*-FP8, -NVFP4 | Llama, format 1 | 1.7-11.4 GB | 64/64⁶ | ●² | no encoder | ● |
| basal-1.0-{1.5B,4.5B} (+FP8, NVFP4) | Llama, format 1 | 1.0-9.5 GB | 64/64⁶ | ●² | no encoder | ● |

The live columns are option orders 1, 2 and 2le4 (or 1 and 2), on the 53 text requests of `tools/compare.py`.

1. The repository holds 19 GB: a second copy of the weights in `think/` for the think mode. `basal-lite serve` skips
   it unless `--think` is given.
2. NVIDIA ModelOpt checkpoints (FP8, NVFP4). vLLM needs sm80 for FP8 and sm75 for NVFP4, so an RTX 3090 (sm86) runs
   them through Marlin kernels (8 / 4-bit weights, bf16 activations); probabilities may differ slightly. The client
   picks the calibration as `basal-serve --mode vllm` does: `CALIBRATION.vllm.json` (omni FP8), else
   `CALIBRATION.json`. `CALIBRATION.fp8.json` belongs to basal's own FP8 engine, not to vLLM.
3. A hybrid model (recurrent layers). vLLM's prefix cache shares a long state between questions (5 questions over 4,400
   tokens: 75% of prompt tokens from the cache), but short states (about 100 tokens) never hit the cache: vLLM sets a
   544-token block for this hybrid and reuses full blocks only. The block size follows from the recurrent state size
   and cannot be lowered.
4. The 12B and 31B templates add an empty `<|channel>thought\n<channel|>` block before the answer when thinking is
   off, and strip it from an assistant turn, so a prefilled answer (`continue_final_message`) never gets it. Without
   the fix the prompts matched 0/69; `template.py` adds the block when the assistant turn is the last one: 69/69.
5. `basal-serve --mode vllm --media on` refuses Gemma 4 Unified models because vLLM reads their images and audio
   differently from the model's processor. basal-lite refuses them the same way. Text works.
6. Format 1: 10 option codes and two option orders by default. 5 of the 69 vectors have more than 10 options and are
   refused, as by basal.
7. basal-1.5 bf16 ships `evidence_head.pt` (pointing at the part of the state behind an answer). basal-lite cannot
   serve it: it needs the model's hidden states, which vLLM's API does not return.

## Where the decisions differ

Every difference found live is a tie or a threshold edge, never a systematic shift:

- **Ties between options.** Synthetic vectors with 26-255 near-identical options ("Category N", 1-5% each), two
  multi labels at 0.800 and 0.808, or an `act` question that defers on both sides and only its `answer` field differs
  at 0.53 vs 0.47.
- **Threshold edges.** A multi label at 0.483 in basal and 0.500 in lite against a 0.5 threshold; `--orders auto`
  where the first readout sits at the 0.8 band edge, so one side reads a second order and the other does not.
- **Think mode, force.** The same think server, but greedy reasoning in vLLM depends on which requests share a batch,
  so two of 52 reasonings diverge after 50-100 tokens and end in a different answer.

The prompt length (`usage.input_tokens`) matched on every request that both sides served, which is the direct check
that both sent the same prompt. With `--context-overflow truncate`, the truncated states matched to the token
(`head_tokens`, `tail_tokens`, `dropped_tokens`).

## Which model on a 24 GB card

- Ready to use: basal-2.0-mini, basal-2.0-4.5B, basal-omni-1.0-mini (text and media), basal-omni-1.0-mini-FP8,
  basal-omni-1.0-max-FP8 (12B, text only), basal-omni-pico (0.8B, fits next to other GPU services),
  basal-2.0-max-FP8 (11B, the most accurate 2.0), basal-2.0-4.5B-FP8 (5 GB), basal-2.0-mini-FP8 (2 GB).
- Too large: the bf16 max and ultra models and ultra-FP8. They need a 48-80 GB card or two GPUs.
- basal-1.x works but is older and weaker than 2.0; it only makes sense for existing deployments.
