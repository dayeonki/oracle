import faiss
import os
import numpy as np
from datetime import datetime
from glob import glob
from time import time


def train(train_npy_file: str, index_type: str, output_dir: str, reference_npy_files: str):
    train_data = np.load(train_npy_file)
    if not (0.999 < sum(train_data[0] ** 2) < 1.001):
        raise ValueError(f"Normalize train data of {train_npy_file}")

    n_data, dim = train_data.shape
    train_time = time()
    index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
    index.train(train_data)
    train_time = time() - train_time
    print(f"Train time {train_time:.1f} sec with {n_data} data")

    for i_data, reference_npy_file in enumerate(reference_npy_files):
        ref_data = np.load(reference_npy_file)
        if not (0.999 < sum(ref_data[0] ** 2) < 1.001):
            raise ValueError(f"Normalize train data of {reference_npy_file}")
        index.add(ref_data)
        now = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"*** [{now}] Add {i_data + 1} / {len(reference_npy_files)} ***")
    
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, f"{output_dir}/{index_type}_trained")
    print(f"*** Saved trained index at `{output_dir}/{index_type}_trained` ***")


def main():
    nlist = 8192
    pq_nbits = 64
    index_type = f"IVF{nlist},PQ{pq_nbits}"

    train_npy_file = "npy/paracrawl/bimnli12/enfr.paracrawl.en.00.01000000.npy"
    output_dir = "indexer/cnli/en"
    reference_npy_files = sorted(glob("npy/paracrawl/bimnli12/enfr.paracrawl.en.0*.**000000.npy"))
    
    for reference_npy_file in reference_npy_files:
        print(f" - {reference_npy_file}")
    train(train_npy_file=train_npy_file, index_type=index_type, output_dir=output_dir, reference_npy_files=reference_npy_files)


if __name__ == "__main__":
    main()