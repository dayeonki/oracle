import numpy as np
import torch
import hnswlib


embedding_file = "pt"
data_file = "txt"


def load_data(embedding_file, data_file):
    embeddings = torch.load(embedding_file)
    embeddings = embeddings.cpu().detach().numpy()
    embeddings = embeddings.astype(np.float32)
    indices = np.arange(embeddings.shape[0])
    with open(data_file) as f:
        texts = [line.strip() for line in f]
    assert len(texts) == embeddings.shape[0]
    return embeddings, indices, texts


def incremental_indexing(embeddings, indices, texts, index_time_param=16, search_time_param=50):
    """
    Sample output
        array([[0.0000000e+00, 2.3841858e-07, 4.6016252e-01, 4.6772724e-01,
                4.7063535e-01, 4.7199458e-01, 4.7199458e-01, 4.7873682e-01,
                4.7876978e-01, 4.8590457e-01]], dtype=float32)

        [QUERY]: We were exposed to the latest theories on child care.
        [NEIGHBOR]: We were exposed to the latest theories on child care.
        [NEIGHBOR]: I was at last relieved of my children's care.
        [NEIGHBOR]: We were in the process of giving birth to our first child.
        [NEIGHBOR]: Heard the child abduction case news not long ago.
        [NEIGHBOR]: We found the children to be undernourished.
        [NEIGHBOR]: We beguiled the children with fairy tales.
        [NEIGHBOR]: A children's ward has recently been added to this hospital.
        [NEIGHBOR]: We left the baby in the care of our neighbour.
        [NEIGHBOR]: More recently, researchers have grown particularly concerned about the adverse effects that cell phone usage could have on children.
        [NEIGHBOR]: I have been witness to the increasing weight problems our children are facing.
    """
    index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
    index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=index_time_param)
    index.add_items(embeddings[:300000], indices[:300000])
    index.set_ef(search_time_param)
    print(index.element_count)

    assert index.element_count == 300000  # Indexed data count

    # Find for each query in embedding
    for i in range(index.element_count):
        labels, distances = index.knn_query(embeddings[i].reshape(1, -1), k=10)
        print(distances)
        print(f"[QUERY]: {texts[0]}")
        for idx in labels[0]:
            print(f"[NEIGHBOR]: {texts[idx]}")


def main():
    embeddings, indices, texts = load_data(embedding_file, data_file)
    index = incremental_indexing(embeddings, indices, texts)


if __name__ == "__main__":
    main()