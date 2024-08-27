from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default="MLP-KTLim/llama-3-Korean-Bllossom-8B")
parser.add_argument('--local_dir', default="./bllossom/")
args = parser.parse_args()

def main(args):
    snapshot_download(repo_id=args.model_id, local_dir=args.local_dir,revision="main")


if __name__ == '__main__':
    main(args)
