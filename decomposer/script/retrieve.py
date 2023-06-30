import argparse
import numpy as np
import torch
import json
import importlib
from utils import *
from sklearn.metrics import pairwise_distances_argmin
from transformers import AutoModel, AutoTokenizer


def pairwise_similarity(src_embed, tgt_embed):
    src_embed = src_embed.cpu().detach().numpy()
    tgt_embed = tgt_embed.cpu().detach().numpy()
    similarity = pairwise_distances_argmin(src_embed, tgt_embed, metric="cosine")
    accuracy = []
    for i, p in enumerate(similarity):
        accuracy.append(i == p)
    return np.mean(accuracy) * 100, similarity


def retrieve(args, device, tokenizer, model, pooler):
    src_sentences = [line.strip() for line in open(args.src_data_path)]
    tgt_sentences = [line.strip() for line in open(args.tgt_data_path)]
    
    src_semantic_embed = semantic_embedding(tokenizer, model, pooler, src_sentences, args.batch_size, device)
    tgt_semantic_embed = semantic_embedding(tokenizer, model, pooler, tgt_sentences, args.batch_size, device)

    src_lang_embed = language_embedding(tokenizer, model, pooler, src_sentences, args.batch_size, device)
    tgt_lang_embed = language_embedding(tokenizer, model, pooler, tgt_sentences, args.batch_size, device)

    accuracy_semantic, prediction_semantic = pairwise_similarity(src_semantic_embed, tgt_semantic_embed)
    accuracy_language, prediction_language = pairwise_similarity(src_lang_embed, tgt_lang_embed)
    
    results = []
    for i in range(len(src_sentences)):
        result = {
            "src_lang": args.src_lang,
            "tgt_lang": args.tgt_lang,
            "semantic_acc": accuracy_semantic,
            "language_acc": accuracy_language,
            "src_text": src_sentences[i],
            "tgt_text": tgt_sentences[i],
            "prediction_idx": int(prediction_semantic[i]),
            "prediction": tgt_sentences[prediction_semantic[i]]
        }
        results.append(result)
    print(f"Average accuracy: {accuracy_semantic}")

    # Save results
    with open(args.save_path, "w", encoding="utf-8") as file:
        for result in results:
            json.dump(result, file, ensure_ascii=False)
            file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="sentence-transformers/LaBSE")
    parser.add_argument("-p", "--pooler_path", default="../models/dream_labse_jako_boc.pt")
    parser.add_argument("-s", "--save_path", default="../results/jako_dream_labse_boc.jsonl")
    
    parser.add_argument("-n", "--n_languages", type=int, default=2)
    parser.add_argument("-b", "--batch_size", default=512)
    parser.add_argument("-d", "--decomposer_type", type=str, default="dream")
    
    parser.add_argument("-sd", "--src_data_path", default="../data/bitext/jako/jako.ja")
    parser.add_argument("-td", "--tgt_data_path", default="../data/bitext/jako/jako.ko")
    parser.add_argument("-sl", "--src_lang", default="ja")
    parser.add_argument("-tl", "--tgt_lang", default="ko")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(device)

    embedding_size = model.embeddings.word_embeddings.embedding_dim
    PoolerClass = getattr(importlib.import_module(f"model.{args.decomposer_type}"), "MLP")
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=args.n_languages)
    pooler.load_state_dict(torch.load(args.pooler_path))
    pooler.to(device)

    retrieve(args, device, tokenizer, model, pooler)


if __name__ == "__main__":
    main()
