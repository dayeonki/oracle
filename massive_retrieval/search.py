import faiss
import math
import numpy as np
import os
import json
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
    ref_text_name: str = "reference_trans",
    query_text_name: str = "query_trans",
    k: int = 100,
    batch_size: int = 10000,
    similarity_threshold: float = 0.0,
    nprobe: int = 1,
    dataset_variation: str = "bimnli12",
    start_language: str = "fr",
):
    ref_text_files = sort_files(ref_text_files)
    query_npy_files = sort_files(query_npy_files)
    query_text_files = sort_files(query_text_files)

    assert len(query_npy_files) == len(query_text_files)

    loading_time = time()
    ref_texts = loadlines_multifile(ref_text_files)
    
    index = faiss.read_index(index_file)
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    print("Number of index: ", index.ntotal)
    print("Length of reference texts: ", len(ref_texts))
    assert index.ntotal == len(ref_texts)
    
    loading_time = time() - loading_time
    print(f"*** Load index from {index_file} ({loading_time:.3f} sec) ***")
    print(f"- n reference data: {index.ntotal}")
    print(f"- nprobe: {nprobe}\n")

    os.makedirs(f"{output_dir}/{dataset_variation}/{start_language}", exist_ok=True)
    for query_npy_file, query_text_file in zip(query_npy_files, query_text_files):
        query_npy = np.load(query_npy_file)
        query_texts = loadlines(query_text_file)
        assert query_npy.shape[0] == len(query_texts)
        
        basename = os.path.basename(query_text_file)
        if basename.endswith(".lfs"):
            basename = basename[:-4]

        outputs = []; similarity = []
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
                print(output)
                outputs.append(output)
                similarity.append(sim)
        print("Total mean of similarity: ", sum(similarity) / len(similarity))
        
        output_file = f"{output_dir}/{dataset_variation}/{start_language}/{basename}.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            for each in outputs:
                f.write(f"{json.dumps(each, ensure_ascii=False)}\n")


def main():
    index_file = "massive_retrieval/indexer/bimnli12/fr/IVF8192,PQ64_trained"
    output_dir = "massive_retrieval/result"
    query_npy_files = ["npy/paracrawl/bimnli12/enfr.paracrawl.en.00.02000000.npy"]
    query_text_files = ["data/bitext/enfr/paracrawl/enfr.paracrawl.en.00.01"]
    ref_text_files = ["data/bitext/enfr/paracrawl/enfr.paracrawl.fr.00.lfs",
                      "data/bitext/enfr/paracrawl/enfr.paracrawl.fr.01.lfs",
                      "data/bitext/enfr/paracrawl/enfr.paracrawl.fr.02.lfs",
                      "data/bitext/enfr/paracrawl/enfr.paracrawl.fr.03.lfs"]
    search(
        index_file=index_file,
        output_dir=output_dir,
        query_npy_files=query_npy_files,
        query_text_files=query_text_files,
        ref_text_files=ref_text_files,
    )


if __name__ == "__main__":
    main()
