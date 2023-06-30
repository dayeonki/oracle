import numpy as np
import torch
import json
import hnswlib
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2LMHeadModel


device = 0
model_path = "path/to/enko_gpt"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval().to("cuda:0")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

embedding_file = "enko_gpt/embedded/train.enko.en.emb.300000.pt"
data_file = "enko_gpt/data/split/enko.en.train.300000"

model.eval()


def load_data(embedding_file, data_file):
    embeddings = torch.load(embedding_file)
    embeddings = embeddings.cpu().detach().numpy()
    embeddings = embeddings.astype(np.float32)
    indices = np.arange(embeddings.shape[0])
    with open(data_file) as f:
        texts = [line.strip() for line in f]
    assert len(texts) == embeddings.shape[0]
    return embeddings, indices, texts


def incremental_indexing(embeddings, indices, texts, num_neighbors, index_time_param=16, search_time_param=50):
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

    assert index.element_count == 300000  # Indexed data count

    # Find for each query in embedding
    neighbor_jsonls = []
    redundancy_rate = []
    for i in trange(index.element_count):
        labels, distances = index.knn_query(embeddings[i].reshape(1, -1), k=num_neighbors+1)    
        distances = distances.tolist()    
        neighbors = []
        for idx in labels[0][1:]:
            neighbors.append(texts[idx])
        neighbor_jsonl = {
            "distances": distances,
            "query": texts[i],
            "neighbors": neighbors
        }
        neighbor_jsonls.append(neighbor_jsonl)

        # Check for same first token
        query_emb = tokenizer(texts[i], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)["input_ids"]
        query_emb = query_emb.cpu().detach().numpy()

        count = 0
        for neighbor in neighbors:
            neighbor_embed = tokenizer(neighbor, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)["input_ids"]
            neighbor_embed = neighbor_embed.cpu().detach().numpy()
            if query_emb[0][0] == neighbor_embed[0][0]:
                count += 1
            else:
                pass
        redundancy_rate.append(count / num_neighbors)
    print(f"Redundancy Rate (k={num_neighbors}): ", (sum(redundancy_rate) / len(redundancy_rate)) * 100, "%")

    # Save file
    with open(f'neighbors/hnsw_en_{num_neighbors}.jsonl', 'w', encoding='utf-8') as f:
        for neighbor_jsonl in neighbor_jsonls:
            json.dump(neighbor_jsonl, f, ensure_ascii=False)
            f.write("\n")


def main():
    # Set number of neighbors
    num_neighbors_list = [1,2,3,4,5,10,20]
    embeddings, indices, texts = load_data(embedding_file, data_file)
    for num_neighbors in num_neighbors_list:
        incremental_indexing(embeddings, indices, texts, num_neighbors)


if __name__ == "__main__":
    main()