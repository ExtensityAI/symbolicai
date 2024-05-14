import argparse

from transformers import AutoModelForCausalLM

# Model candidates: 
# - EleutherAI/gpt-j-6B
# - EleutherAI/gpt-neo-125M
# - EleutherAI/gpt-neo-1.3B
# - EleutherAI/gpt-neo-2.7B
# - EleutherAI/gpt-neox-20b
# - togethercomputer/GPT-JT-6B-v1
# - gpt2
# - gpt2-medium
# - gpt2-large
# - gpt2-xl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument("--cache_dir", type=str, default="/system/user/publicwork/dinu/remote/tmp/cache")
    args = parser.parse_args()

    # download model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    del model


