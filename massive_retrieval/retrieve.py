import argparse
import faiss
import json
import math
import numpy as np
import os
from time import time
from typing import List
from tqdm import trange
from utils import *


def search(
    index_file: str,
    output_dir: str,
    query_npy_files: List[str],
    query_text_files: List[str],
    ref_text_files: List[str],
    ref_tag_files: List[str] = None,
    query_tag_files: List[str] = None,
    ref_text_name: str = "reference_trans",
    ref_tag_name: str = "reference_text",
    query_text_name: str = "query_trans",
    query_tag_name: str = "query_text",
    k: int = 100,
    batch_size: int = 10000,
    similarity_threshold: float = 0.9,
    nprobe: int = 1,
):
    ref_text_files = sort_files(ref_text_files)
    ref_tag_files = sort_files(ref_tag_files)
    query_npy_files = sort_files(query_npy_files)
    query_text_files = sort_files(query_text_files)
    query_tag_files = sort_files(query_tag_files)
    
    if query_tag_files is None:
        query_tag_files = [None for _ in query_text_files]
    assert len(query_npy_files) == len(query_text_files) == len(query_tag_files)

    loading_time = time()
    ref_texts = loadlines_multifile(ref_text_files)
    ref_tags = loadlines_multifile(ref_tag_files)
    
    index = faiss.read_index(index_file)
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    assert index.ntotal == len(ref_texts) and equal_length(ref_texts, ref_tags)
    
    loading_time = time() - loading_time
    print(f"Load index from {index_file} ({loading_time:.3f} sec)")
    print(f"- n reference data: {index.ntotal}")
    print(f"- nprobe: {nprobe}\n")

    os.makedirs(f"{output_dir}/fold", exist_ok=True)
    
    for query_npy_file, query_text_file, query_tag_file in zip(query_npy_files, query_text_files, query_tag_files):
        query_npy = np.load(query_npy_file)
        query_texts = loadlines(query_text_file)
        query_tags = loadlines(query_tag_file)
        assert query_npy.shape[0] == len(query_texts)
        
        if query_tags is not None:
            assert len(query_texts) == len(query_tags)
        basename = os.path.basename(query_text_file)
        
        if basename.endswith(".lfs"):
            basename = basename[:-4]
        output_file = f"{output_dir}/fold/{basename}.jsonl"

        outputs = []
        n_batches = math.ceil(query_npy.shape[0] / batch_size)
        for batch_idx in trange(n_batches, desc=f"Querying {basename}"):
            begin, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
            query = query_npy[begin:end]
            retrieve_sim, retrieve_indices = index.search(query, k)

            batch_col, batch_row = np.where(retrieve_sim >= similarity_threshold)
            output_sim = retrieve_sim[batch_col, batch_row]
            query_indices = batch_col + (batch_idx * batch_size)
            ref_indices = retrieve_indices[batch_col, batch_row]

            for query_idx, ref_idx, sim in zip(query_indices, ref_indices, output_sim):
                output = {
                    "query_idx": int(query_idx),
                    "reference_idx": int(ref_idx),
                    "similarity": float(sim),
                    query_text_name: query_texts[query_idx],
                    ref_text_name: ref_texts[ref_idx],
                }
                if ref_tags is not None:
                    output[ref_tag_name] = ref_tags[ref_idx]
                if query_tags is not None:
                    output[query_tag_name] = query_tags[query_idx]
                outputs.append(output)
        
        with open(output_file, "a", encoding="utf-8") as f:
            for each in outputs:
                f.write(f"{json.dumps(each, ensure_ascii=False)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    search(
        index_file=config["index_file"],
        output_dir=config["output_dir"],
        query_npy_files=config["query_npy_files"],
        query_text_files=config["query_text_files"],
        ref_text_files=config["ref_text_files"],
        nprobe=config["nprobe"],
        k=config["k"],
        ref_tag_files=None,
        query_tag_files=None,
        similarity_threshold=-1,
    )


if __name__ == "__main__":
    main()
