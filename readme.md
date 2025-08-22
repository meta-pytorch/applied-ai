
### Applied AI repo
For experiments and research on Applied AI.

### Projects

#### Kernels

Housing a variety of Triton and CUDA kernels for training and inference.

Inference kernels = no backward pass support.

##### Triton Kernels

#### 1 - Triton - MoE (Mixtral) GEMM for accelerating inference. Uses col major access pattern to increase locality.

<img width="556" alt="moe_gemm_a100" src="https://github.com/meta-pytorch/applied-ai/assets/46302957/9eece843-b5e1-4250-a98a-3ae79dff1bc3">


#### 2 - Triton - Fused Softmax for both training and inference.

<img width="556" alt="softmax_fused" src="https://github.com/meta-pytorch/applied-ai/assets/46302957/de11686b-4c17-4696-857a-4f56488d6df3">

#### 3 - Triton - Fused RMSNorm for both training and inference. 
[Fused RMSNorm Kernel](https://github.com/meta-pytorch/applied-ai/blob/main/kernels/triton/training/rms_norm/fused_rms_norm.py)

#### Other projects from Applied AI

1. [CUDA Mode](https://github.com/cuda-mode) - Reading group for learning CUDA programming - ([Discord](https://discord.gg/cudamode), [Lecture Materials](https://github.com/cuda-mode/lectures), [Lecture recordings](https://www.youtube.com/@CUDAMODE))
2. [llama-recipes](https://github.com/meta-llama/llama-recipes) - Recipes for fine-tuning and inference for Llama model series
3. NeurIPS'23 [LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io/) - 1LLM + 1GPU + 1Day competition - ([website](https://llm-efficiency-challenge.github.io/), [code](https://github.com/llm-efficiency-challenge), [NeurIPS Workshop recordings](https://neurips.cc/virtual/2023/competition/66594))

### Papers and Publications

1. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation [paper](https://pytorch.org/assets/pytorch2-2.pdf)
2. Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with SplitK Work Decomposition [paper](https://ai.meta.com/research/publications/accelerating-a-triton-fused-kernel-for-w4a16-quantized-inference-with-splitk-work-decomposition/)
3. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel [paper](https://arxiv.org/abs/2304.11277)
4. Sustainable AI: Environmental Implications, Challenges and Opportunities [paper](https://arxiv.org/abs/2111.00364)



### License
The applied-ai repo is released under the [BSD 3](LICENSE) license.
