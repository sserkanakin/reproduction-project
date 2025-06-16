FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git jq ninja-build && rm -rf /var/lib/apt/lists/*

ENV CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
    TORCH_CUDA_ARCH_LIST=8.9 \
    PYTHONUNBUFFERED=1

# 1️⃣  Build flash-attn from source (10 min on L4)
RUN pip install --upgrade pip && \
    pip install --no-build-isolation flash-attn==2.5.9.post1 \
        --extra-index-url https://pypi.org/simple

RUN pip uninstall -y flash_attn flash_attn_2_cuda && \
     pip install flash-attn==2.5.5 \




# 2️⃣  BitsAndBytes for 4-bit LoRA
RUN pip install bitsandbytes==0.43.1

# 3️⃣  Latest LLaVA
RUN pip install "git+https://github.com/haotian-liu/LLaVA.git@main"

# 4️⃣  Helpers
RUN pip install transformers==4.40.0 peft==0.10.0 accelerate==0.29.3 datasets tqdm

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
