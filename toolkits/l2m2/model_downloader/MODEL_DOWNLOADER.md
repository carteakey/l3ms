# Dynamic Model Downloader

A flexible and configurable tool for downloading models from Hugging Face Hub with support for patterns, custom directories, batch downloads, and machine-readable size estimates.

## Features

- **Dynamic Configuration**: Use JSON config files or command-line arguments
- **Pattern Matching**: Include/exclude specific file patterns (e.g., `*Q8*`, `*.bin`)
- **Batch Downloads**: Download multiple models in one go
- **Enable/Disable Models**: Toggle models on/off without removing them from config
- **Auto-directory Generation**: Automatically organize models by repository structure
- **Supported Download Backend**: Reports `hf_xet` when available and otherwise uses the `huggingface_hub` default transport
- **Incremental Sync**: `--update` pulls only changed/new files for already-downloaded models
- **Dry-run Estimates**: Reports filtered, cache-aware byte totals as machine-readable JSON without downloading model files
- **Error Handling**: Robust error handling with detailed feedback
- **Download Throttling**: Control concurrency with `--slow` or `--max-workers`

## Installation

From the repository root, install the downloader dependencies:

```bash
python -m pip install -r requirements-downloader.txt
```

The script uses `hf_xet` when it is installed and supported by `huggingface_hub`. Otherwise, it uses the Hub client's default download path. The former `hf_transfer` backend is deprecated and is not required.

## Usage

### 1. Single Model Download

Download a single model with command-line arguments:

```bash
# Basic download
python model_downloader/download_hf_model.py --repo-id microsoft/DialoGPT-medium

# Download with specific patterns
python model_downloader/download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q6_K*"

# Download to custom directory
python model_downloader/download_hf_model.py --repo-id microsoft/DialoGPT-medium --local-dir ./my_models/dialog

# Download specific revision/branch
python model_downloader/download_hf_model.py --repo-id microsoft/DialoGPT-medium --revision main

# Slow preset (equivalent to --max-workers 4)
python model_downloader/download_hf_model.py --repo-id microsoft/DialoGPT-medium --slow

# Throttle bandwidth/parallelism by lowering workers
python model_downloader/download_hf_model.py --repo-id microsoft/DialoGPT-medium --max-workers 2

# Force re-download existing files
python model_downloader/download_hf_model.py --repo-id microsoft/DialoGPT-medium --force-download
```

### 2. Batch Download with Configuration

Create a configuration file and download multiple models:

```bash
# Use existing config
python model_downloader/download_hf_model.py --config model_downloader/models_config.json

# Sync updates for already-downloaded models
python model_downloader/download_hf_model.py --config model_downloader/models_config.json --update

# Throttle concurrency
python model_downloader/download_hf_model.py --config model_downloader/models_config.json --slow
```

### 3. Configuration File Format

```json
{
  "base_models_dir": "./models",
  "models": [
    {
      "enabled": true,
      "repo_id": "microsoft/DialoGPT-medium",
      "allow_patterns": ["*.bin", "*.json", "*.txt"],
      "description": "DialoGPT medium conversational model"
    },
    {
      "enabled": true,
      "repo_id": "Qwen/Qwen3-32B-GGUF",
      "local_dir": "./models/qwen/Qwen3-32B-GGUF",
      "allow_patterns": ["*Q6_K*"],
      "ignore_patterns": ["*.md"],
      "max_workers": 2,
      "description": "Qwen3 32B with Q6_K quantization"
    },
    {
      "enabled": false,
      "repo_id": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
      "allow_patterns": ["*Q8*"],
      "description": "Qwen3 30B Instruct - disabled, won't download"
    }
  ]
}
```

### 4. Machine-readable Download Estimate

Use `--estimate-json` with either a repository or a configuration file to ask Hugging Face Hub for filtered dry-run metadata. It does not download model files.

```bash
# Estimate one filtered repository
python model_downloader/download_hf_model.py --estimate-json \
  --repo-id hf-internal-testing/tiny-random-gpt2 \
  --allow-patterns "config.json"

# Estimate all eligible models in a configuration
python model_downloader/download_hf_model.py --estimate-json \
  --config model_downloader/models_config.json
```

Successful output is exactly one compact JSON object on stdout, with no progress messages or backend banner:

```json
{"schema_version":1,"models":[{"repo_id":"hf-internal-testing/tiny-random-gpt2","revision":"main","matched_files":1,"total_bytes":807,"download_bytes":807,"cached_bytes":0}],"totals":{"models":1,"matched_files":1,"total_bytes":807,"download_bytes":807,"cached_bytes":0}}
```

The schema contains:

- `schema_version`: currently `1`
- `models`: one summary per estimated repository, including the resolved revision, number of matched files, total matched bytes, bytes still requiring download, and bytes already cached
- `totals`: the same counts and byte fields aggregated across all returned models

The estimate honors repository, revision, local directory, allow/ignore patterns, and force-download selection. In configuration mode, disabled entries and entries without a valid `repo_id` are skipped. Pattern values may be a string or an array of strings. Worker settings do not affect byte totals.

Cache status is evaluated for the selected local directory at the time of the request. A file contributes to `cached_bytes` only when the Hub reports it cached and not scheduled for download; a file scheduled for transfer contributes to `download_bytes`. Estimates are bounded to 256 eligible models and 10,000 matched files.

This mode is intended for programs rather than interactive output. On failure, stdout is empty, the process exits nonzero, and stderr contains one bounded error line.

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--repo-id` | `-r` | Repository ID to download |
| `--local-dir` | `-d` | Local directory to save the model |
| `--allow-patterns` | `-a` | File patterns to include |
| `--ignore-patterns` | `-i` | File patterns to exclude |
| `--config` | `-c` | Path to JSON configuration file |
| `--revision` | | Specific branch/tag/commit to pin |
| `--update` | `-u` | Sync updates for models already on disk; skips fresh downloads |
| `--force-download` | | Re-download existing files (overrides incremental check) |
| `--max-workers` | | Max concurrent download workers |
| `--slow` | | Slow preset (`max_workers=4`) |
| `--base-models-dir` | | Base directory for all models |
| `--estimate-json` | | Emit one machine-readable dry-run estimate without downloading model files |

## Configuration Options

### Model Configuration

Each model in the configuration can have these properties:

- `enabled` (optional, default: `true`): Set to `false` to skip this model without removing it
- `repo_id` (required): Hugging Face repository ID
- `local_dir` (optional): Custom local directory path
- `allow_patterns` (optional): File pattern string or list of patterns to include
- `ignore_patterns` (optional): File pattern string or list of patterns to exclude
- `revision` (optional): Specific git revision/branch/tag
- `force_download` (optional): Whether to re-download existing files
- `max_workers` (optional): Max concurrent file downloads
- `description` (optional): Human-readable description

### Pattern Examples

Common file patterns for different model types:

- **GGUF Models**: `["*Q4_K_M*", "*Q6_K*", "*Q8_0*"]`
- **PyTorch Models**: `["*.bin", "*.pt", "*.safetensors"]`
- **Config Files**: `["*.json", "*.txt", "config.yaml"]`
- **Exclude Documentation**: `["*.md", "*.gitattributes", "README*"]`

## Directory Structure

By default, models are organized as:

```
models/
├── microsoft/
│   └── DialoGPT-medium/
├── qwen/
│   ├── Qwen3-32B-GGUF/
│   └── Qwen3-30B-A3B-Instruct-2507-GGUF/
└── ggml-org/
    └── gpt-oss-120b-GGUF/
```

## Examples

### Download Specific Quantizations

```bash
# Download only Q8 quantized models
python model_downloader/download_hf_model.py --repo-id unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --allow-patterns "*Q8*"

# Download multiple quantization levels
python model_downloader/download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q4_K_M*" "*Q6_K*"
```

### Exclude Large Files

```bash
# Skip documentation and large unquantized files
python model_downloader/download_hf_model.py --repo-id microsoft/DialoGPT-medium --ignore-patterns "*.md" "*F32*" "*F16*"
```

### Custom Organization

```bash
# Download to specific directory structure
python model_downloader/download_hf_model.py --repo-id microsoft/DialoGPT-medium --local-dir ./conversations/dialog-medium
```

### Enable/Disable Models in Config

You can temporarily disable models in your configuration without deleting or commenting them out:

```json
{
  "models": [
    {
      "enabled": true,
      "repo_id": "microsoft/DialoGPT-medium",
      "description": "This will download"
    },
    {
      "enabled": false,
      "repo_id": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
      "description": "This will be skipped"
    },
    {
      "repo_id": "Qwen/Qwen3-32B-GGUF",
      "description": "No 'enabled' field means enabled by default"
    }
  ]
}
```

This is useful for:

- Testing with a subset of models
- Temporarily skipping large downloads
- Keeping model configurations for future use
- Managing download priorities

## Troubleshooting

### Common Issues

1. **Import Error**: Install the root downloader requirements with `python -m pip install -r requirements-downloader.txt`
2. **Permission Errors**: Check write permissions for the target directory
3. **Network Issues**: Verify internet connection and Hugging Face Hub access
4. **Disk Space**: Ensure sufficient disk space for large models

### Environment Variables

- `HF_TOKEN`: Your Hugging Face token for private repositories

If `HF_HUB_ENABLE_HF_TRANSFER=1` remains in an older environment, unset it. When `hf_xet` is unavailable, the script reports that legacy `hf_transfer` setting as deprecated; current downloads use `hf_xet` when available or the `huggingface_hub` default transport.

### Debugging

Add verbose output by checking the console messages. The script provides detailed feedback about:
- Download progress
- Pattern matching
- Error conditions
- Final file locations

## Migration from Static Script

If you're migrating from the old static script, you can:

1. Create a config file with your existing models:

```json
{
  "models": [
    {
      "repo_id": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
      "local_dir": "/home/kchauhan/Desktop/repos/lllms/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF",
      "allow_patterns": ["*Q8*"]
    }
  ]
}
```

2. Run with the config:

```bash
python model_downloader/download_hf_model.py --config your_models.json
```

This provides the same functionality with much more flexibility for future use.
