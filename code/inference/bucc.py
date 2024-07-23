import numpy as np
import argparse
import torch
import importlib
from transformers import AutoModel, AutoTokenizer
from bucc_utils import *
from utils import *
from scipy import stats


def bucc_f1(labels, predictions):
    labels = set([tuple(l.split("\t")) for l in labels])
    predictions = set([tuple(l.split("\t")) for l in predictions])
    ncorrect = len(labels.intersection(predictions))
    if ncorrect > 0:
        prec = ncorrect / len(predictions)
        rec = ncorrect / len(labels)
        f1_val = 2 * prec * rec / (prec + rec)
    else:
        prec = rec = f1_val = 0
    return {"F1-score": f1_val * 100, "Precision": prec * 100, "Recall": rec * 100}


def retrieve_bucc(args, model, tokenizer, pooler, device):
    def process_text(name):
        all_text,all_idx,id2text = [],[],{}
        fout = open(name + ".txt", "w")
        for line in open(name):
            idx, text = line.strip().split("\t")
            all_idx.append(idx.strip())
            all_text.append(text.strip())
            id2text[idx] = text.strip()
            fout.write(idx.strip()+"\n")
        fout.close()
        return all_idx, all_text, id2text

    data_dir = "data/bucc2018/"
    all_f1_semantic = []
    all_f1_language = []
    for lg in ["fr","ru","zh","de"]:
        src_file = data_dir + f"{lg}-en.dev.{lg}"
        tgt_file = data_dir + f"{lg}-en.dev.en"
        gold_file = data_dir + f"{lg}-en.dev.gold"

        src_id, src_text, src_id2text = process_text(src_file)
        src_semantic_embed = semantic_embedding(tokenizer, model, pooler, src_text, args.batch_size, device).cpu().detach().numpy()
        src_language_embed = language_embedding(tokenizer, model, pooler, src_text, args.batch_size, device).cpu().detach().numpy()

        tgt_id, tgt_text, tgt_id2text = process_text(tgt_file)
        tgt_semantic_embed = semantic_embedding(tokenizer, model, pooler, tgt_text, args.batch_size, device).cpu().detach().numpy()
        tgt_language_embed = language_embedding(tokenizer, model, pooler, tgt_text, args.batch_size, device).cpu().detach().numpy()

        sem_candidate_file = data_dir + f"{lg}-{args.pooler_nickname}-candidate.sem.tsv"
        lang_candidate_file = data_dir + f"{lg}-{args.pooler_nickname}-candidate.lang.tsv"
        mine_bitext(src_semantic_embed, tgt_semantic_embed, src_file+".txt", tgt_file+".txt", sem_candidate_file)
        mine_bitext(src_language_embed, tgt_language_embed, src_file+".txt", tgt_file+".txt", lang_candidate_file)

        sem_candidate2score = {}
        for line in open(sem_candidate_file):
            sem_candidate = line.strip().split("\t")
            sem_candidate2score[tuple(sem_candidate[1:])] = float(sem_candidate[0]) 
        
        lang_candidate2score = {}
        for line in open(lang_candidate_file):
            lang_candidate = line.strip().split("\t")
            lang_candidate2score[tuple(lang_candidate[1:])] = float(lang_candidate[0]) 

        gold_target = [line.strip() for line in open(gold_file)]

        sem_threshold = bucc_optimize(sem_candidate2score, gold_target)
        semantic_pred = []
        for sem_cand in sem_candidate2score:
            if sem_candidate2score[sem_cand] >= sem_threshold:
                semantic_pred.append("\t".join(sem_cand))

        lang_threshold = bucc_optimize(lang_candidate2score, gold_target)
        language_pred = []
        for lang_cand in lang_candidate2score:
            if lang_candidate2score[lang_cand] >= lang_threshold:
                language_pred.append("\t".join(lang_cand))
        
        semantic_result = bucc_f1(gold_target, semantic_pred)
        language_result = bucc_f1(gold_target, language_pred)
        
        all_f1_semantic.append(semantic_result["F1-score"])
        all_f1_language.append(language_result["F1-score"])
        
        print(f"F1 for {lg} (Semantic):", semantic_result)
        print(f"F1 for {lg} (Language):", language_result)
    
    print("BUCC F1 (Semantic):", np.mean(all_f1_semantic))
    print("BUCC F1 (Language):", np.mean(all_f1_language))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str)
    parser.add_argument("--pooler_nickname", type=str)
    parser.add_argument("-p", "--pooler_path", type=str)

    parser.add_argument("-d", "--decomposer_type", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-n", "--n_languages", type=int, default=13)
    args = parser.parse_args()

    print("POOLER: ", args.pooler_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name_or_path == "/fs/clip-scratch/dayeonki/project/acl_oracle/ckpt/mSimCSE":
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", cache_dir="/fs/clip-scratch/dayeonki/.cache")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir="/fs/clip-scratch/dayeonki/.cache")
    model = AutoModel.from_pretrained(args.model_name_or_path, cache_dir="/fs/clip-scratch/dayeonki/.cache").to(device)

    embedding_size = model.embeddings.word_embeddings.embedding_dim
    PoolerClass = getattr(importlib.import_module(f"model.{args.decomposer_type}"), "MLP")
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=args.n_languages)
    pooler.load_state_dict(torch.load(args.pooler_path))
    pooler.to(device)

    retrieve_bucc(args, model, tokenizer, pooler, device)


if __name__ == "__main__":
    main()