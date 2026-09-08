# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id = "ggml-org/gpt-oss-120b-GGUF",
#     local_dir = "/home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF",
#     # allow_patterns = ["*F16*"],
# )

# snapshot_download(
#     repo_id = "Qwen/Qwen3-32B-GGUF",
#     local_dir = "/home/kchauhan/Desktop/repos/lllms/models/qwen/Qwen3-32B-GGUF",
#     allow_patterns = ["*Q6_K*"],
# )

snapshot_download(
    repo_id = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    local_dir = "/home/kchauhan/Desktop/repos/lllms/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF",
    allow_patterns = ["*Q8*"],
)
