# LLM_eval_data

## Create environment

```bash
# with conda 
conda create -n llm_eval python=3.11
conda activate llm_eval
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# with mamba
mamba create -n llm_eval python=3.11
mamba activate llm_eval
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Install dependencies

```bash
pip install requirements.txt
```

## Install llama-cpp-python

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```