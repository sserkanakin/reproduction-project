# Dockerized MMIU LLaVA Evaluation Pipeline

This setup allows you to run the MMIU LLaVA evaluation pipeline in a Docker container, ensuring a consistent environment for reproducible results, especially when moving to a cloud environment with GPUs.

## Prerequisites

* **Docker:** Installed on your system. [Install Docker](https://docs.docker.com/get-docker/)
* **NVIDIA Container Toolkit:** (If using NVIDIA GPUs on a Linux host) Installed to allow Docker containers to access host GPUs. [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* **MMIU Dataset JSON:** Your MMIU dataset file in JSON format (e.g., `to-data.json` as seen in your project structure). This file should contain the questions, image paths (relative to an image base directory), options, and answers for the "temporal ordering" task.
* **MMIU Images Directory:** A root directory on your host machine that contains all the MMIU images referenced in your JSON file (e.g., the `Continuous-temporal` folder and its contents).
* **`.env` File:** Create a file named `.env` (e.g., inside your `eval-pipeline` directory as shown in your structure). This file should contain your OpenAI API key:
    ```env
    OPENAI_API_KEY="sk-yourActualOpenAiKeyGoesHere"
    # Optional: If you use a proxy or different base URL for OpenAI
    # OPENAI_BASE_URL="your_openai_proxy_url"
    ```
    **Important:** Ensure this `.env` file (or the directory containing it if you prefer to keep secrets separate) is listed in your `.gitignore` file to prevent committing secrets.

## Project Structure (Assumed for `docker run` commands)

Based on your screenshot, when running `docker build .` and `docker run ...` from the `reproduction-project` root:

reproduction-project/├── Dockerfile├── requirements.txt├── .gitignore├── README.md│├── eval-pipeline/│   ├── .env                          # Your OpenAI API key file│   ├── eval-llava.py             # Assumed to be your main pipeline script│   └── data/│       ├── to-data.json          # Your MMIU task data JSON│       └── Continuous-temporal/  # Root of your MMIU images for this task│           └── temporal_ordering/│               └── image1.jpg│               └── ...│└── eval_results/                 # Output directory (will be created on host if mapped)*(Other files like `run-inference.py` are noted but the `Dockerfile` currently uses `eval-pipeline/eval-llava.py` as the source for `run_mmiu_llava_pipeline.py` in the image).*

## Building the Docker Image

1.  **Navigate to Project Root:** Open your terminal in the `reproduction-project` directory (where your `Dockerfile` is located).
2.  **Ensure `requirements.txt` is Correct:**
    * Make sure the PyTorch installation line in `requirements.txt` is uncommented and corresponds to the CUDA version in your `Dockerfile`'s base image (e.g., for `nvidia/cuda:12.1.1-...`, use the PyTorch build for `cu121`).
3.  **Build the Image:**
    ```bash
    docker build -t mmiu-llava-pipeline .
    ```
    The `.` indicates that the build context is the current directory.

If you modify `requirements.txt` or `eval-pipeline/eval-llava.py` (which is copied into the image), you'll need to rebuild the image.

## Running the Docker Container

You will mount your local data directories, the `.env` file, and an output directory into the Docker container at runtime.

**Example `docker run` command (run from `reproduction-project` root):**

Adjust paths if your local structure differs slightly from the example above.

```bash
docker run --rm \
    # For GPU support (if your Dockerfile uses an NVIDIA base image and you have NVIDIA drivers/toolkit):
    --gpus all \
    # Mount your .env file (read-only is good practice for secrets)
    -v "$(pwd)/eval-pipeline/.env:/app/.env:ro" \
    # Mount your MMIU data JSON file (e.g., to-data.json)
    -v "$(pwd)/eval-pipeline/data/to-data.json:/app/data/mmiu_task_data.json:ro" \
    # Mount the base directory containing all MMIU images for the task.
    # If images are in 'eval-pipeline/data/Continuous-temporal/...',
    # and your JSON has paths like './Continuous-temporal/...',
    # then mounting 'eval-pipeline/data/' is correct.
    -v "$(pwd)/eval-pipeline/data/:/app/mmiu_images_root/:ro" \
    # Mount a directory for saving results (read-write)
    -v "$(pwd)/eval_results:/app/eval_results" \
    # Optional: Mount Hugging Face cache for persistence
    # Create ~/.cache/huggingface on your host if it doesn't exist
    # -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    # Name of the Docker image you built
    mmiu-llava-pipeline \
    # --- Script Arguments for run_mmiu_llava_pipeline.py (inside the container) ---
    --mmiu_data_path "/app/data/mmiu_task_data.json" \
    --image_base_dir "/app/mmiu_images_root/" \
    --output_dir "/app/eval_results" \
    --llava_model_id "llava-hf/llava-interleave-qwen-7b-hf" \ # Target 7B model for cloud
    # --use_quantization # Enable if your cloud GPU has limited VRAM and you want 8-bit
    # --device "cuda" # Usually auto-detected correctly with --gpus all
    --verbose 1 \
    # --max_samples 10 # Useful for quick testing
    # Add any other arguments your script needs
Explanation of docker run options:--rm: Automatically remove the container when it exits.--gpus all: (Optional) Exposes all available host GPUs to the container. Requires NVIDIA Container Toolkit on Linux.-v "$(pwd)/eval-pipeline/.env:/app/.env:ro": Mounts your local .env file (located in reproduction-project/eval-pipeline/) into /app/.env inside the container.-v "$(pwd)/eval-pipeline/data/to-data.json:/app/data/mmiu_task_data.json:ro": Mounts your MMIU data JSON file.-v "$(pwd)/eval-pipeline/data/:/app/mmiu_images_root/:ro": Mounts your local eval-pipeline/data/ directory (which contains Continuous-temporal/ etc.) to /app/mmiu_images_root/ inside the container. The script's --image_base_dir "/app/mmiu_images_root/" will then correctly resolve relative image paths from your JSON (e.g., if JSON has ./Continuous-temporal/..., it becomes /app/mmiu_images_root/Continuous-temporal/...).-v "$(pwd)/eval_results:/app/eval_results": Mounts a local directory named eval_results (it will be created in your reproduction-project directory if it doesn't exist) to /app/eval_results inside the container. This is where output files are saved.mmiu-llava-pipeline: The name of the Docker image you built.The lines after the image name are the command-line arguments passed to run_mmiu_llava_pipeline.py (which is eval-pipeline/eval-llava.py copied into the image). Note how paths for data and output now refer to their locations inside the container.Important Considerations for Cloud Deployment:Base Image in Dockerfile: The FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 line is for NVIDIA GPUs. Adjust if your cloud instance is CPU-only or uses different GPU types (e.g., AMD, AWS Inferentia/Trainium, Google TPUs), and update requirements.txt for PyTorch accordingly.Model Caching (Hugging Face Cache):For Cloud Environments (Recommended): Mount a persistent disk from your cloud provider to /root/.cache/huggingface inside the container (as shown commented out in the docker run example). This avoids re-downloading models on every container start.Alternative (Bake into Image): Uncomment and adapt the model download lines in your Dockerfile. This makes the Docker image much larger but ensures the model is always present.Data Location in the Cloud:Transfer your MMIU data JSON and image directories to your cloud instance's filesystem. Then, adjust the host-side paths in the docker run -v ... commands to point to these locations on the cloud VM.For more advanced setups, your Python script could be modified to read directly from cloud storage (e.g., S3, GCS).Resource Allocation: Ensure your cloud instance has sufficient RAM, CPU cores, and GPU VRAM for the 7B LLaVA model.