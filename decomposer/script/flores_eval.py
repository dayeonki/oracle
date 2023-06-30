import argparse
import numpy as np
import torch
import json
import importlib
from glob import glob
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import pairwise_distances_argmin


def embedding(tokenizer, base_model, pooler, sentences, batch_size, device):
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for i in range(-(-len(sentences) // batch_size)):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]
        encoded = tokenizer(sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = base_model(**encoded.to(device)).pooler_output
        embeddings = pooler(outputs)[0]
        all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_embeddings = torch.stack(all_embeddings)
    return all_embeddings


def pairwise_similarity(src_embed, tgt_embed):
    src_embed = src_embed.cpu().detach().numpy()
    tgt_embed = tgt_embed.cpu().detach().numpy()
    similarity = pairwise_distances_argmin(src_embed, tgt_embed, metric="cosine")
    accuracy = []
    for i, p in enumerate(similarity):
        accuracy.append(i == p)
    return np.mean(accuracy) * 100, similarity


def flores_eval(tokenizer, model, pooler, data_path, save_path, batch_size, device):    
    files = []
    for file in sorted(glob(data_path)):
        files.append(file)

    results = []
    for i in range(len(files)):
        for j in range(len(files)-1):
            source_file = files[i]
            source_text = [line.strip() for line in open(source_file)]
            target_file = files[j+1]
            target_text = [line.strip() for line in open(target_file)]

            src_emb = embedding(tokenizer, model, pooler, source_text, batch_size, device)
            tgt_emb = embedding(tokenizer, model, pooler, target_text, batch_size, device)

            acc, pred = pairwise_similarity(src_emb, tgt_emb)
            print(f'{source_file[-12:-4]} - {target_file[-12:-4]}:', round(acc * 100, 2))

            result = {
                "source": source_file[-12:-4],
                "target": target_file[-12:-4],
                "acc": round(acc * 100, 2)
                }
            results.append(result)

    with open(save_path, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="sentence-transformers/LaBSE")
    parser.add_argument("-d", "--data_path", type=str)
    parser.add_argument("-s", "--save_path", default="../results/jako_dream_labse_boc.jsonl")
    
    parser.add_argument("-n", "--n_languages", type=int, default=2)
    parser.add_argument("-b", "--batch_size", default=512)
    parser.add_argument("--decomposer_type", type=str, default="dream")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(device)
    
    embedding_size = model.embeddings.word_embeddings.embedding_dim
    PoolerClass = getattr(importlib.import_module(f"model.{args.decomposer_type}"), "MLP")
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=args.n_languages)
    pooler.load_state_dict(torch.load(args.model_path))
    pooler.to(device)

    flores_eval(tokenizer, model, pooler, args.data_path, args.save_path, args.batch_size, device)


if __name__ == "__main__":
    main()
