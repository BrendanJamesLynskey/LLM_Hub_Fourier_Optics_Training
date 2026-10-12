# Fourier Optics for Training

Could a Fourier-optical engine, which computes a 2-D Fourier transform (and with a mask, a convolution) in one pass of light, accelerate **LLM training**? This mini-series is the companion to [Fourier Optics for Inference](https://github.com/BrendanJamesLynskey/LLM_Hub_Fourier_Optics_Inference). Training adds a backward pass, weights that change every step and gradients that need their own precision. Three interactive decks work through what that changes: where transforms appear in a training step, with an **Amdahl analysis of training-step FLOP shares** from a published, tested script; how optics itself is trained (in-situ backpropagation, physics-aware training); and where an optical engine could sit in a training cluster.

**Live index:** https://brendanjameslynskey.github.io/LLM_Hub_Fourier_Optics_Training/

## Presentations in this series

| # | Title | Status | What it covers |
|---|-------|--------|----------------|
| 01 | [Transforms in Training Workloads](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/) | live | What the backward pass does to a Fourier-domain layer: the gradient of a convolution is a correlation, so dX reuses the filter's spectrum but the weight gradient multiplies two data-dependent spectra. Training-step Amdahl analysis for transformer, Hyena-2, hybrid and block-circulant variants; the share a once-per-step mask could hold; the mask-rewrite bill when weights change every step. |
| 02 | [Optical Backpropagation and In-Situ Training](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/) | live | In-silico training of diffractive networks, in-situ backpropagation by the adjoint method, physics-aware training, direct feedback alignment on a photonic co-processor, analogue noise and the reality gap, and the ENOB a gradient format asks for. |
| 03 | [Optical Engines in a Training Cluster](https://brendanjameslynskey.github.io/FOptTrn_03_Optical_Training_Clusters/) | live | Data, tensor and pipeline parallelism with an optical transform engine: step time against mask-rewrite time, layer-major scheduling, the optimizer step and frequency-domain gradient compression (DeMo). |

## Analysis and code

| Path | What's inside |
|------|---------------|
| [analysis/train_flop_share.py](analysis/train_flop_share.py) | The training-step FLOP-share (Amdahl) analysis. It uses a Llama-3-8B-shaped model in four variants (transformer, Hyena-2, hybrid, block-circulant), forward and backward, with the Fourier-domain work split into weight-masked and data-by-data products. Also: a cross-check against the 6N rule; DeMo's transform cost; SLM frames per optimizer step; the ENOB needed to match BF16, FP8 E4M3 and FP8 E5M2; NumPy checks of the backward formulas |
| [analysis/results.md](analysis/results.md) | Its output: the source of every number in the decks |
| [analysis/test_train_model.py](analysis/test_train_model.py) | pytest: spectral gradients match direct correlations and finite differences, the transformer matches the 6N rule, and backward is about twice forward |
| [js/train_flop_model.js](js/train_flop_model.js) | The JavaScript port of the ledger that runs deck 01's training-step Amdahl explorer |
| [analysis/test_js_parity.py](analysis/test_js_parity.py) | pytest: the JavaScript port matches the Python to 1e-12 for every variant, sequence length and step size tested (needs `node`) |

Reproduce: `pip install numpy pytest`, then `python analysis/train_flop_share.py` and `pytest analysis` (the parity test needs `node`). The script is self-contained. It extends the forward-pass ledger of the inference series' [flop_share.py](https://github.com/BrendanJamesLynskey/LLM_Hub_Fourier_Optics_Inference/blob/main/analysis/flop_share.py) and keeps the same model shape and FLOP conventions.

## Results

- **Training does not rescue the transform share.** For Hyena-2 at 4,096 tokens, transforms are 0.21% of a forward pass and 0.15% of a training step. The backward pass has as many FFTs as the forward, but the matmuls double.
- **Block-circulant weights**: the step share is 12.2% (Amdahl bound 1.14x). A fifth of that optical work is data-by-data (the weight gradient), so a once-per-step mask holds only 9.7%.
- **Masks change every step.** On the illustrative 2-megapixel, 1,031 Hz device, rewriting the weight spectra takes 0.5–2 s per step, comparable to or longer than the digital step. Writing data-by-data masks per sequence would take minutes on one device, and still about a whole digital step (Hyena-2) or many steps (block-circulant) spread over one engine per GPU.
- **Precision**: matching FP8 E5M2 gradients needs about 5.3–5.8 ENOB, and BF16 about 10.3–10.7.
- **DeMo's** DCTs are about 5×10⁻⁶ of a step's FLOPs. Their value is in link bytes saved, not compute.

## What is illustrative

The FLOP counts are exact for the stated conventions. The following are **illustrative** and labelled where they are used:

- the step size (about 1M tokens);
- the 64-GPU, 400 TFLOP/s cluster used for step times;
- the 2-megapixel, 1,031 Hz mask device;
- the hybrid ratio and the circulant block size;
- the gradient spike model.

A block-circulant LLM at this scale is **speculative**.

## Glossary: concepts and where they are explained

The full glossary, with anchors, is on the [live index](https://brendanjameslynskey.github.io/LLM_Hub_Fourier_Optics_Training/#glossary). Optics and inference concepts link to the [Fourier Optics for Inference glossary](https://brendanjameslynskey.github.io/LLM_Hub_Fourier_Optics_Inference/#glossary) instead of being repeated.

| Concept | What it is | Where it is explained |
|---|---|---|
| The 6N rule | About 6 FLOPs per parameter per token to train a dense model, plus an attention term | [FOptTrn 01 · the 6N rule, reproduced](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/#slide-02); results.md §1 |
| The gradient of a convolution is a correlation | dU = IFFT(G·conj(H)), dH = IFFT(G·conj(U)) | [FOptTrn 01 · the gradient of a convolution is a correlation](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/#slide-04); [FOptTrn 01 · three 4f passes](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/#slide-05); results.md §6 |
| Weight-masked and data-by-data products | Fourier-domain products with a weight operand (a mask per step) or two data operands (a mask per sequence) | [FOptTrn 01 · weight-masked and data-by-data products](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/#slide-06); [FOptTrn 03 · data-by-data products stay digital](https://brendanjameslynskey.github.io/FOptTrn_03_Optical_Training_Clusters/#slide-08); results.md §2 |
| Static share and the training Amdahl bound | Transforms plus weight-masked products, as a share of a step | [FOptTrn 01 · the training-step Amdahl table](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/#slide-08); [FOptTrn 01 · interactive Amdahl explorer](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/#slide-10); [FOptTrn 03 · the static Amdahl bound](https://brendanjameslynskey.github.io/FOptTrn_03_Optical_Training_Clusters/#slide-09); results.md §2 |
| Mask rewrites per optimizer step | Weight spectra rewritten after every update, for forward and backward | [FOptTrn 01 · masks rewritten every step](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/#slide-11); [FOptTrn 03 · masks change every step](https://brendanjameslynskey.github.io/FOptTrn_03_Optical_Training_Clusters/#slide-04); [FOptTrn 03 · interactive: step time against mask time](https://brendanjameslynskey.github.io/FOptTrn_03_Optical_Training_Clusters/#slide-05); results.md §4 |
| Diffractive deep neural networks | Passive diffractive layers trained in simulation, then fabricated | [FOptTrn 02 · in-silico training and the reality gap](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-03) |
| In-situ backpropagation (adjoint method) | Measuring gradients in a photonic circuit by sending the error backwards | [FOptTrn 02 · the adjoint method](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-04); [FOptTrn 02 · in-situ backpropagation, demonstrated](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-05) |
| Physics-aware training | Physical forward pass, digital-twin backward pass | [FOptTrn 02 · physics-aware training](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-06); [FOptTrn 02 · interactive: three trainers](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-11) |
| Direct feedback alignment | Random-projection error feedback, done optically | [FOptTrn 02 · direct feedback alignment](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-07) |
| The reality gap | Mismatch between the trained model and the hardware | [FOptTrn 02 · in-silico training and the reality gap](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-03); [FOptTrn 02 · interactive: three trainers](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-11) |
| Gradient precision and FP8 E5M2 | The ENOB a gradient format asks of the ADC | [FOptTrn 02 · precision for training formats](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-10); results.md §5 |
| Analogue noise in a training loop | Zero-mean noise as regulariser, biased error as a fault | [FOptTrn 02 · analogue noise in a training loop](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-09); [FOptTrn 02 · interactive: three trainers](https://brendanjameslynskey.github.io/FOptTrn_02_Optical_Backpropagation/#slide-11) |
| Layer-major scheduling | One mask write per layer per step, against pipeline parallelism | [FOptTrn 01 · layer-major order](https://brendanjameslynskey.github.io/FOptTrn_01_Transforms_in_Training/#slide-12); [FOptTrn 03 · layer-major scheduling against the pipeline](https://brendanjameslynskey.github.io/FOptTrn_03_Optical_Training_Clusters/#slide-06) |
| DeMo (decoupled momentum) | DCT-compressed momentum exchange between workers | [FOptTrn 03 · the optimizer step and DeMo](https://brendanjameslynskey.github.io/FOptTrn_03_Optical_Training_Clusters/#slide-11); results.md §3 |

## Where this fits

Part of the [LLMs hub](https://github.com/BrendanJamesLynskey/LLMs), under *Hardware & Inference*, and indexed from [Hardware](https://github.com/BrendanJamesLynskey/Hardware). **Sister series:** [Fourier Optics for Inference](https://github.com/BrendanJamesLynskey/LLM_Hub_Fourier_Optics_Inference) (the physics, and the inference case), [Compute in Transit](https://github.com/BrendanJamesLynskey/LLM_Hub_Compute_in_Transit) (computing on training traffic in the link, including DeMo), [FHE Accelerator Simulators](https://github.com/BrendanJamesLynskey/FHE_Hub_Accelerator_Simulators) and the [Simulation Engineering Toolkit](https://github.com/BrendanJamesLynskey/SimEng_Hub_Toolkit).

## Licence

Slides and text: [CC BY 4.0](LICENSE). Code: [MIT](LICENSE-CODE).
