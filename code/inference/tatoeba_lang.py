import numpy as np
import argparse
import torch
import importlib
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import cosine
from utils import *


def pairwise_similarity(src_embed, tgt_embed):
    src_embed = src_embed.cpu().detach().numpy()
    tgt_embed = tgt_embed.cpu().detach().numpy()
    similarity = pairwise_distances_argmin(src_embed, tgt_embed, metric="cosine")
    distances = []
    accuracy = []
    for i, p in enumerate(similarity):
        accuracy.append(i == p)
        retrieved_instance = tgt_embed[p]
        distance = cosine(src_embed[i], retrieved_instance)
        distances.append(distance)
    return np.mean(accuracy) * 100, similarity, np.mean(distances)


def retrieve_tatoeba(model, tokenizer, pooler, batch_size, device):
    data_dir = "data/tatoeba/"
    all_accuracies1, all_accuracies2, all_accuracies14_lgen, all_accuracies14_enlg = [],[],[],[]
    
    language_dict = {
        "ara":"ar", "heb":"he", "vie":"vi", "ind":"id",
        "jav":"jv", "tgl":"tl", "eus":"eu", "mal":"ml",
        "tel":"te", "afr":"af", "nld":"nl", "eng":"en", "deu":"de",
        "ell":"el", "ben":"bn", "hin":"hi", "mar":"mr", "urd":"ur",
        "tam":"ta", "fra":"fr", "ita":"it", "por":"pt", "spa":"es",
        "bul":"bg", "rus":"ru", "jpn":"ja", "kat":"ka", "kor":"ko",
        "tha":"th", "swa":"sw", "zho":"zh", "kaz":"kk", "tur":"tr",
        "est":"et", "fin":"fi", "hun":"hu", "fas":"fa", "aze": "az",
        "lit": "lt","pol": "pl", "ukr": "uk", "ron": "ro"
    }

    language_dict_shortened = {l2: l3 for l3, l2 in language_dict.items()}
    lg_list14_1 = "ar bg de el ".split()
    lg_list14_2 = "zh fr hi ru es sw th tr ur vi".split()
    lg_list36_1 = "af bn".split()
    lg_list36_2 = "et eu fi he hu it jv ja ka kk ko ml mr nl fa pt ta te tl".split()
    lg_list1 = "ar de".split()
    lg_list2 = "es fr it ja nl pt ro zh".split()

    for lg2 in lg_list14_1:
        lg = language_dict_shortened[lg2]

        f1 = f"{data_dir}tatoeba-test-v2023-09-26.{lg}-eng.txt"
        sources = []
        targets = []
        with open(f1, "r", encoding="utf-8") as file:
            for line in file:
                columns = line.strip().split("\t")
                source = columns[2] if len(columns) > 2 else None
                target = columns[3] if len(columns) > 3 else None
                sources.append(source)
                targets.append(target)

        sources = sources[:1000]
        targets = targets[:1000]
        
        assert len(sources) == len(targets)
        emb1 = language_embedding(tokenizer, model, pooler, sources, batch_size, device)
        emb2 = language_embedding(tokenizer, model, pooler, targets, batch_size, device)

        acc1, pred1, distance1 = pairwise_similarity(emb1, emb2)
        acc2, pred2, distance2 = pairwise_similarity(emb2, emb1)
        print(f"{lg} <-> eng accuracy: ", acc1)
        print(f"eng <-> {lg} accuracy: ", acc2)
        print(f"{lg} - eng average accuracy: ", (acc1+acc2) / 2)
        print("\n===================\n")
        all_accuracies14_lgen.append(acc1)
        all_accuracies14_enlg.append(acc2)
    
    for lg2 in lg_list14_2:
        lg = language_dict_shortened[lg2]

        f1 = f"{data_dir}tatoeba-test-v2023-09-26.eng-{lg}.txt"
        sources = []
        targets = []
        with open(f1, "r", encoding="utf-8") as file:
            for line in file:
                columns = line.strip().split("\t")
                source = columns[2] if len(columns) > 2 else None
                target = columns[3] if len(columns) > 3 else None
                sources.append(source)
                targets.append(target)

        sources = sources[:1000]
        targets = targets[:1000]
        
        assert len(sources) == len(targets)
        emb1 = language_embedding(tokenizer, model, pooler, sources, batch_size, device)
        emb2 = language_embedding(tokenizer, model, pooler, targets, batch_size, device)

        acc1, pred1, distance1 = pairwise_similarity(emb1, emb2)
        acc2, pred2, distance2 = pairwise_similarity(emb2, emb1)
        print(f"{lg} <-> eng accuracy: ", acc2)
        print(f"eng <-> {lg} accuracy: ", acc1)
        print(f"{lg} - eng average accuracy: ", (acc1+acc2) / 2)
        print("\n===================\n")
        all_accuracies14_lgen.append(acc2)
        all_accuracies14_enlg.append(acc1)

    all_accuracies36_lgen = all_accuracies14_lgen.copy()
    all_accuracies36_enlg = all_accuracies14_enlg.copy()
 
    for lg2 in lg_list36_1:
        lg = language_dict_shortened[lg2]

        f1 = f"{data_dir}tatoeba-test-v2023-09-26.{lg}-eng.txt"
        sources = []
        targets = []
        with open(f1, "r", encoding="utf-8") as file:
            for line in file:
                columns = line.strip().split("\t")
                source = columns[2] if len(columns) > 2 else None
                target = columns[3] if len(columns) > 3 else None
                sources.append(source)
                targets.append(target)

        sources = sources[:1000]
        targets = targets[:1000]
        
        assert len(sources) == len(targets)
        emb1 = language_embedding(tokenizer, model, pooler, sources, batch_size, device)
        emb2 = language_embedding(tokenizer, model, pooler, targets, batch_size, device)

        acc1, pred1, distance1 = pairwise_similarity(emb1, emb2)
        acc2, pred2, distance2 = pairwise_similarity(emb2, emb1)
        print(f"{lg} <-> eng accuracy: ", acc1)
        print(f"eng <-> {lg} accuracy: ", acc2)
        print(f"{lg} - eng average accuracy: ", (acc1+acc2) / 2)
        print("\n===================\n")
        all_accuracies36_lgen.append(acc1)
        all_accuracies36_enlg.append(acc2)
    
    for lg2 in lg_list36_2:
        lg = language_dict_shortened[lg2]

        f1 = f"{data_dir}tatoeba-test-v2023-09-26.eng-{lg}.txt"
        sources = []
        targets = []
        with open(f1, "r", encoding="utf-8") as file:
            for line in file:
                columns = line.strip().split("\t")
                source = columns[2] if len(columns) > 2 else None
                target = columns[3] if len(columns) > 3 else None
                sources.append(source)
                targets.append(target)

        sources = sources[:1000]
        targets = targets[:1000]
        
        assert len(sources) == len(targets)
        emb1 = language_embedding(tokenizer, model, pooler, sources, batch_size, device)
        emb2 = language_embedding(tokenizer, model, pooler, targets, batch_size, device)

        acc1, pred1, distance1 = pairwise_similarity(emb1, emb2)
        acc2, pred2, distance2 = pairwise_similarity(emb2, emb1)
        print(f"{lg} <-> eng accuracy: ", acc2)
        print(f"eng <-> {lg} accuracy: ", acc1)
        print(f"{lg} - eng average accuracy: ", (acc1+acc2) / 2)
        print("\n===================\n")
        all_accuracies36_lgen.append(acc2)
        all_accuracies36_enlg.append(acc1)

    # Alphabets that come before e
    for lg2 in lg_list1:
        lg = language_dict_shortened[lg2]

        f1 = f"{data_dir}tatoeba-test-v2023-09-26.{lg}-eng.txt"
        sources = []
        targets = []
        with open(f1, "r", encoding="utf-8") as file:
            for line in file:
                columns = line.strip().split("\t")
                source = columns[2] if len(columns) > 2 else None
                target = columns[3] if len(columns) > 3 else None
                sources.append(source)
                targets.append(target)

        sources = sources[:1000]
        targets = targets[:1000]
        
        assert len(sources) == len(targets)
        emb1 = language_embedding(tokenizer, model, pooler, sources, batch_size, device)
        emb2 = language_embedding(tokenizer, model, pooler, targets, batch_size, device)

        acc1, pred1, distance1 = pairwise_similarity(emb1, emb2)
        acc2, pred2, distance2 = pairwise_similarity(emb2, emb1)
        print(f"{lg} <-> eng accuracy: ", acc1)
        print(f"eng <-> {lg} accuracy: ", acc2)
        print(f"{lg} - eng average accuracy: ", (acc1+acc2) / 2)
        print("\n===================\n")
        all_accuracies1.append(acc1)
        all_accuracies2.append(acc2)

    # Alphabets that come after e
    for lg2 in lg_list2:
        lg = language_dict_shortened[lg2]

        f1 = f"{data_dir}tatoeba-test-v2023-09-26.eng-{lg}.txt"
        sources = []
        targets = []
        with open(f1, "r", encoding="utf-8") as file:
            for line in file:
                columns = line.strip().split("\t")
                source = columns[2] if len(columns) > 2 else None
                target = columns[3] if len(columns) > 3 else None
                sources.append(source)
                targets.append(target)
        
        sources = sources[:1000]
        targets = targets[:1000]
        
        assert len(sources) == len(targets)
        emb1 = language_embedding(tokenizer, model, pooler, sources, batch_size, device)
        emb2 = language_embedding(tokenizer, model, pooler, targets, batch_size, device)

        acc1, pred1, distance1 = pairwise_similarity(emb1, emb2)
        acc2, pred2, distance2 = pairwise_similarity(emb2, emb1)
        print(f"{lg} <-> eng accuracy: ", acc2)
        print(f"eng <-> {lg} accuracy: ", acc1)
        print(f"{lg} - eng average accuracy: ", (acc1+acc2) / 2)
        print("\n===================\n")
        all_accuracies1.append(acc2)
        all_accuracies2.append(acc1)

    print("Average Accuracy 14 (XX-EN): ", np.mean(all_accuracies14_lgen))
    print("Average Accuracy 14 (EN-XX): ", np.mean(all_accuracies14_enlg))
    print("Average Accuracy 36 (XX-EN): ", np.mean(all_accuracies36_lgen))
    print("Average Accuracy 36 (EN-XX): ", np.mean(all_accuracies36_enlg))
    print("Average Accuracy 10 (XX-EN): ", np.mean(all_accuracies1))
    print("Average Accuracy 10 (EN-XX): ", np.mean(all_accuracies2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--pooler_path", type=str)
    parser.add_argument("--decomposer_type", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("--n_languages", type=int, default=13)
    args = parser.parse_args()
    
    print("POOLER: ", args.pooler_path)
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

    retrieve_tatoeba(model, tokenizer, pooler, args.batch_size, device)


if __name__ == "__main__":
    main()