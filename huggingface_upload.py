from huggingface_hub import HfApi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="llama-3-Korean-Bllossom-8B-Q5_K_M")
parser.add_argument('--file_path', default="llama-3-Korean-Bllossom-8B-Q5_K_M.gguf")
args = parser.parse_args()

def main(args):
    api = HfApi()

    model_id = "ywhwang/" + args.model_name
    api.create_repo(model_id, exist_ok=True, repo_type="model")
    api.upload_file(
        path_or_fileobj=args.file_path,
        path_in_repo=args.file_path,
        repo_id=model_id,
    )

if __name__ == '__main__':
    main(args)