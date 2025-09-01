#!/usr/bin/env python3
"""
Script to upload SPIRAL trained models to Hugging Face Hub.
Handles CPU-only loading and direct file upload to avoid GPU memory issues.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional, List
import shutil
from datetime import datetime

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from transformers import AutoTokenizer, AutoConfig
except ImportError:
    print("Please install required packages:")
    print("pip install huggingface_hub transformers")
    exit(1)


def get_available_checkpoints(models_dir: str) -> List[str]:
    """Get list of available model checkpoints."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    checkpoints = []
    for item in models_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            # Check if it has the required model files
            if (item / "config.json").exists() and (item / "model.safetensors.index.json").exists():
                checkpoints.append(item.name)
    
    return sorted(checkpoints)


def create_model_card(
    repo_name: str,
    base_model: str,
    training_config: dict,
    checkpoint_step: str,
    model_type: str = "qwen3"
) -> str:
    """Create a model card for the uploaded model."""
    
    if "octothinker" in repo_name.lower():
        model_name = "SPIRAL OctoThinker-3B Multi-Agent Model"
        model_size = "3B parameters"
        base_model_display = "OctoAI/OctoThinker-3B"
    else:
        model_name = "SPIRAL Qwen3-8B Multi-Agent Model"
        model_size = "8B parameters"
        base_model_display = "Qwen/Qwen3-8B-Base"
    
    card_content = f"""---
base_model: {base_model}
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- spiral
- self-play
- reinforcement-learning
- {model_type}
- multi-agent
---

# {model_name}

This model was trained using the SPIRAL (Self-Play Iterative Reinforcement learning for Adaptation and Learning) framework.

## Model Details

- **Base Model**: {base_model_display}
- **Training Framework**: SPIRAL
- **Checkpoint**: {checkpoint_step}
- **Model Size**: {model_size}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Training Configuration

The model was trained with self-play on multiple environments:
- KuhnPoker-v1
- TicTacToe-v0  
- SimpleNegotiation-v1

### Training Parameters
```json
{json.dumps(training_config, indent=2)}
```

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained(
    "{repo_name}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## License

This model is licensed under the Apache License 2.0.
"""
    return card_content


def upload_model_checkpoint(
    checkpoint_path: str,
    repo_name: str,
    private: bool = True,
    commit_message: Optional[str] = None
):
    """Upload a model checkpoint to Hugging Face Hub."""
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    print(f"Uploading checkpoint: {checkpoint_path.name}")
    print(f"Repository: {repo_name}")
    
    # Initialize HF API
    api = HfApi()
    
    try:
        # Create repository if it doesn't exist
        print("Creating repository...")
        create_repo(
            repo_id=repo_name,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"Repository created/verified: {repo_name}")
        
        # Load config to extract training info
        config_path = checkpoint_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Determine model type and base model from config
        if config.get("model_type") == "llama":
            base_model = "OctoAI/OctoThinker-3B"
            model_type = "octothinker"
        else:
            base_model = "Qwen/Qwen3-8B-Base"
            model_type = "qwen3"
        
        # Create training config summary
        training_config = {
            "learning_rate": "1e-6",
            "train_batch_size": 128,
            "num_ppo_epochs": 2,
            "temperature": 1.0,
            "max_model_len": 16384,
            "environments": ["KuhnPoker-v1", "TicTacToe-v0", "SimpleNegotiation-v1"],
            "base_model": base_model,
            "framework": "SPIRAL"
        }
        
        # Create model card
        model_card = create_model_card(
            repo_name=repo_name,
            base_model=base_model,
            training_config=training_config,
            checkpoint_step=checkpoint_path.name,
            model_type=model_type
        )
        
        # Write model card to checkpoint directory temporarily
        readme_path = checkpoint_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(model_card)
        
        # Upload the entire folder
        if commit_message is None:
            commit_message = f"Upload SPIRAL {checkpoint_path.name}"
            
        print("Uploading files...")
        upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message
        )
        
        # Clean up temporary README
        if readme_path.exists():
            readme_path.unlink()
            
        print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Upload SPIRAL models to Hugging Face Hub")
    parser.add_argument(
        "--models_dir", 
        type=str,
        default="/root/spiral/oat-output/spiral-spiral-octothinker-3b-multi-8k_0825T07:57:49/saved_models",
        help="Directory containing saved model checkpoints"
    )
    parser.add_argument(
        "--repo_prefix",
        type=str,
        default="spiral-octothinker-3b-multi",
        help="Prefix for HuggingFace repository name"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint to upload (e.g., step_00480). If not specified, will list available checkpoints."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repository private (default: True)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repository public"
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        help="HuggingFace username. If not provided, will use the logged-in user.",
        default="the-acorn-ai"
    )
    
    args = parser.parse_args()
    
    # Handle public flag
    if args.public:
        args.private = False
    
    # Get available checkpoints
    available_checkpoints = get_available_checkpoints(args.models_dir)
    
    if not available_checkpoints:
        print(f"No valid checkpoints found in {args.models_dir}")
        return
    
    print(f"Available checkpoints: {available_checkpoints}")
    
    if not args.checkpoint:
        print("\nAvailable checkpoints:")
        for i, checkpoint in enumerate(available_checkpoints):
            print(f"  {i+1}. {checkpoint}")
        
        choice = input(f"\nSelect checkpoint to upload (1-{len(available_checkpoints)}) or 'all' for all checkpoints: ").strip()
        
        if choice.lower() == 'all':
            checkpoints_to_upload = available_checkpoints
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available_checkpoints):
                    checkpoints_to_upload = [available_checkpoints[idx]]
                else:
                    print("Invalid selection")
                    return
            except ValueError:
                print("Invalid selection")
                return
    else:
        if args.checkpoint in available_checkpoints:
            checkpoints_to_upload = [args.checkpoint]
        elif args.checkpoint == "all":
            checkpoints_to_upload = available_checkpoints
        else:
            print(f"Checkpoint {args.checkpoint} not found in available checkpoints")
            return
    
    # Get HF username
    if not args.hf_username:
        try:
            api = HfApi()
            user_info = api.whoami()
            hf_username = user_info['name']
        except Exception as e:
            print(f"Could not determine HuggingFace username: {e}")
            hf_username = input("Please enter your HuggingFace username: ").strip()
    else:
        hf_username = args.hf_username
    
    # Upload selected checkpoints
    for checkpoint in checkpoints_to_upload:
        checkpoint_path = Path(args.models_dir) / checkpoint
        repo_name = f"{hf_username}/{args.repo_prefix}-{checkpoint.replace('step_', 'step')}"
        
        try:
            upload_model_checkpoint(
                checkpoint_path=str(checkpoint_path),
                repo_name=repo_name,
                private=False
            )
            print(f"✅ {checkpoint} uploaded successfully!")
        except Exception as e:
            print(f"❌ Failed to upload {checkpoint}: {e}")
            continue
    
    print("\n🎉 Upload process completed!")


if __name__ == "__main__":
    main()
