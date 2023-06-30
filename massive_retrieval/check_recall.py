import argparse
import json
import os
import numpy as np
from collections import defaultdict
from glob import glob


def parse_line(line, src_offset):
    data = json.loads(line.strip())
    return data["query_idx"], data["reference_idx"] - src_offset


def check_recalls(src_to_tgt, topk, n_max_hypo=30):
    n_queries = max(src_to_tgt) + 1
    retrieved_indices = -1 * np.ones((n_queries, n_max_hypo), dtype=int)
    for src_idx, tgt_indices in src_to_tgt.items():
        retrieved_indices[src_idx, : len(tgt_indices)] = np.array(tgt_indices)
    true_label = np.arange(n_queries)
    true_label_mat = np.repeat(true_label, n_max_hypo).reshape(n_queries, n_max_hypo)
    check_mat = retrieved_indices - true_label_mat
    recalls = []
    for k in topk:
        n_ok = np.where(check_mat[:, :k] == 0)[0].shape[0]
        recalls.append(n_ok / n_queries)
    return recalls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument("-n", "--n_max_hypo", type=int, default=30)
    args = parser.parse_args()

    topk = [1, 10, 20, 30]
    offset_diff = 1000000
    result_files = sorted(glob(f"{args.experiment_name}/fold/*.jsonl"))
    offset = 0
    for result_file in result_files:
        src_to_tgt = defaultdict(lambda: [])
        with open(result_file) as f:
            for line in f:
                src_idx, tgt_idx = parse_line(line, offset)
                src_to_tgt[src_idx].append(tgt_idx)
        recalls = check_recalls(src_to_tgt, topk, n_max_hypo=30)
        recalls_strf = " | ".join(f"{v:.3f}" for v in recalls)
        print(f"| {os.path.basename(result_file)} | {recalls_strf} |")
        offset += offset_diff


if __name__ == "__main__":
    main()
