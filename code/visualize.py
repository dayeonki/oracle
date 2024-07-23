import numpy as np
import argparse
import torch
import importlib
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from utils import *
from bokeh.io import save
from datavis import scatterplot


def draw_scatterplot(embeds, descriptions, labels, perplexity=10, label_to_color=None):
    tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine", n_iter=300)
    z = tsne.fit_transform(embeds)
    plot = scatterplot(
        x = z[:,0],
        y = z[:,1],
        description=descriptions,
        labels=labels,
        label_to_color=label_to_color
    )
    return plot


def visualize(args, tokenizer, model, pooler, device):
    src_sentences = [line.strip() for line in open(args.src_data_path)]
    tgt_sentences = [line.strip() for line in open(args.tgt_data_path)]

    # Semantic embeddings
    src_semantic_embed = semantic_embedding(tokenizer, model, pooler, src_sentences, args.batch_size, device)
    tgt_semantic_embed = semantic_embedding(tokenizer, model, pooler, tgt_sentences, args.batch_size, device)
    src_semantic_embed = src_semantic_embed.cpu().detach().numpy()
    tgt_semantic_embed = tgt_semantic_embed.cpu().detach().numpy()

    # Language embeddings
    src_lang_embed = language_embedding(tokenizer, model, pooler, src_sentences, args.batch_size, device)
    tgt_lang_embed = language_embedding(tokenizer, model, pooler, tgt_sentences, args.batch_size, device)
    src_lang_embed = src_lang_embed.cpu().detach().numpy()
    tgt_lang_embed = tgt_lang_embed.cpu().detach().numpy()

    # Labels for visualization
    src_label = [args.src_lang+"_mean" for _ in range(len(src_semantic_embed))]
    trg_label = [args.tgt_lang+"_mean" for _ in range(len(tgt_semantic_embed))]

    src_lang_label = [args.src_lang+"_lang" for _ in range(len(src_semantic_embed))]
    trg_lang_label = [args.tgt_lang+"_lang" for _ in range(len(tgt_semantic_embed))]

    src_description = [f"[{i}] {s}" for i, s in enumerate(src_sentences)]
    tgt_description = [f"[{i}] {t}" for i, t in enumerate(tgt_sentences)]

    embeds = np.concatenate([src_semantic_embed, tgt_semantic_embed, src_lang_embed, tgt_lang_embed], axis=0)
    descriptions = src_description + tgt_description + src_description + tgt_description
    labels = np.array(src_label + trg_label + src_lang_label + trg_lang_label)
    
    plot = draw_scatterplot(
        embeds,
        descriptions,
        labels,
        perplexity=10,
        label_to_color={
            f"{args.src_lang}_mean": "#2b83ba", f"{args.tgt_lang}_mean": "#d7191c", 
            f"{args.src_lang}_lang": "#edc858", f"{args.tgt_lang}_lang": "#4bc971"
        }
    )
    
    # Save visualization file
    if args.output_figure is None:
        return
    if not args.output_figure.endswith(".html"):
        args.output_figure = f"{args.output_figure}.html"
    save(plot, args.output_figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="path/to/mSimCSE/result/XLM-R-large-cnli")
    parser.add_argument("-p", "--pooler_path", default="../models/dream_cnli_jako_boc.pt")
    parser.add_argument("-s", "--src_data_path", default="../data/bitext/jako/jako.ja")
    parser.add_argument("-t", "--tgt_data_path", default="../data/bitext/jako/jako.ko")
    
    parser.add_argument("-n", "--n_languages", type=int, default=2)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-d", "--decomposer_type", type=str, default="dream")
    parser.add_argument("-f", "--output_figure", type=str, default="EXAMPLE.html", help="Bokeh figure path")
    
    parser.add_argument("-sl", "--src_lang", type=str, default="ja")
    parser.add_argument("-tl", "--tgt_lang", type=str, default="ko")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(device)

    embedding_size = model.embeddings.word_embeddings.embedding_dim
    PoolerClass = getattr(importlib.import_module("model"), args.decomposer_type)
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=args.n_languages)
    pooler.load_state_dict(torch.load(args.pooler_path))
    pooler.to(device)

    visualize(args, tokenizer, model, pooler, device)


if __name__ == "__main__":
    main()
