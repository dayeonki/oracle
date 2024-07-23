import numpy as np
import argparse
from scipy.stats import spearmanr
import importlib
import torch
from transformers import AutoModel, AutoTokenizer
from utils import *


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def sts_eval(emb1, emb2):
    emb1 = emb1.cpu().detach().numpy()
    emb2 = emb2.cpu().detach().numpy()
    sys_scores = []
    for kk in range(emb2.shape[0]):
        s = cosine(emb1[kk], emb2[kk])
        sys_scores.append(s)
    return sys_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--pooler_path", type=str)
    parser.add_argument("--decomposer_type", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("--n_languages", type=int, default=13)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name_or_path == "ckpt/mSimCSE":
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", cache_dir="")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir="")
    model = AutoModel.from_pretrained(args.model_name_or_path, cache_dir="").to(device)

    embedding_size = model.embeddings.word_embeddings.embedding_dim
    PoolerClass = getattr(importlib.import_module(f"model.{args.decomposer_type}"), "MLP")
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=args.n_languages)
    pooler.load_state_dict(torch.load(args.pooler_path))
    pooler.to(device)

    print("POOLER: ", args.pooler_path)
    sem_results = []
    lang_results = []
    input_dir = "data/sts2017/"

    for name in ["STS.ar-ar", "STS.en-ar", "STS.en-de", "STS.en-en", "STS.en-tr", "STS.es-en", "STS.es-es", "STS.fr-en", "STS.it-en", "STS.nl-en"]:
        print(name)
        sent1 = []
        sent2 = []
        score = []
        with open(input_dir + f"{name}.txt", "r") as file:
            for line in file:
                # Split the line into components using tab as the delimiter
                components = line.strip().split("\t")
                sent1.append(components[0])
                sent2.append(components[1])
                score.append(float(components[2]))
            
            sem_emb1 = semantic_embedding(tokenizer, model, pooler, sent1, args.batch_size, device)
            sem_emb2 = semantic_embedding(tokenizer, model, pooler, sent2, args.batch_size, device)

            lang_emb1 = language_embedding(tokenizer, model, pooler, sent1, args.batch_size, device)
            lang_emb2 = language_embedding(tokenizer, model, pooler, sent2, args.batch_size, device)

            sem_sys_scores = sts_eval(sem_emb1, sem_emb2)
            lang_sys_scores = sts_eval(lang_emb1, lang_emb2)
            sem_result = spearmanr(sem_sys_scores, score)
            lang_result = spearmanr(lang_sys_scores, score)
            print("Semantic: ", sem_result[0] * 100)
            print("Language: ", lang_result[0] * 100)
            sem_results.append(sem_result[0] * 100)
            lang_results.append(lang_result[0] * 100)
        print("============================")
    
    print("Semantic total: ", sum(sem_results) / len(sem_results))
    print("Language total: ", sum(lang_results) / len(lang_results))
    print("============================")


if __name__ == "__main__":
    main()