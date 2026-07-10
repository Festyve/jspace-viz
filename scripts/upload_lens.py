# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Upload a fitted lens to a HuggingFace model repo (requires `hf auth login`).

    python scripts/upload_lens.py \
        --lens lenses/deepseek-coder-1.3b-instruct_jlens_wikitext.pt \
        --repo Festyve/jspace-lenses \
        --path deepseek-coder-1.3b-instruct/lens.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--path", required=True, help="path inside the repo")
    parser.add_argument("--message", default="Upload Jacobian lens")
    args = parser.parse_args()

    from huggingface_hub import create_repo, upload_file

    url = create_repo(args.repo, repo_type="model", exist_ok=True)
    print("repo:", url)
    upload_file(
        repo_id=args.repo,
        path_or_fileobj=args.lens,
        path_in_repo=args.path,
        commit_message=args.message,
    )
    print(f"uploaded {args.lens} -> {args.repo}/{args.path}")


if __name__ == "__main__":
    main()
