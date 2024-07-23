import argparse
import random
import torch
from sklearn.model_selection import train_test_split
import laserembeddings


def embed_and_save(target_language, laser, data_path, embed_path):
    # NOTE: Change the loading code according to your data
    with open(data_path, "r") as f:
        sentences = f.read().rstrip()
        sentences = sentences.split("\n")

    sentences = sentences[:500000]
    languages = []
    for _ in range(500000):
        languages.append(target_language)
    assert len(sentences) == len(languages)
    
    embeddings = laser.embed_sentences(sentences, lang=languages)
    torch.save(embeddings, embed_path)


def train_valid_split(src_emb, tgt_emb, seed, test_size=0.1):
    src_embedding = []
    tgt_embedding = []
    src_embedding.extend(src_emb)
    tgt_embedding.extend(tgt_emb)
    src_embedding = torch.stack(src_embedding)
    tgt_embedding = torch.stack(tgt_embedding)

    train_src_emb, valid_src_emb = train_test_split(src_embedding, test_size=test_size, random_state=seed)
    train_tgt_emb, valid_tgt_emb = train_test_split(tgt_embedding, test_size=test_size, random_state=seed)

    return train_src_emb, valid_src_emb, train_tgt_emb, valid_tgt_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-s", "--seed", type=int, default=42)
    
    parser.add_argument("-sd", "--src_data_path", default="../data/bitext/jako/jako.ja")
    parser.add_argument("-td", "--tgt_data_path", default="../data/bitext/jako/jako.ko")
    parser.add_argument("-se", "--src_embed_path", default="../data/jako.ja.1m.emb.pt")
    parser.add_argument("-te", "--tgt_embed_path", default="../data/jako.ko.1m.emb.pt")
    parser.add_argument("--train_embed_path", default="../data/cnli_train_jako_1m.pt")
    parser.add_argument("--valid_embed_path", default="../data/cnli_valid_jako_1m.pt")
    parser.add_argument("-sl", "--src_lang", type=int, default=0)
    parser.add_argument("-tl", "--tgt_lang", type=int, default=1)
    parser.add_argument("--target_language", type=str)
    args = parser.parse_args()

    laser = laserembeddings.Laser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    random.seed(args.seed)

    # Embed source, target
    embed_and_save(args.target_language, laser, args.src_data_path, args.src_embed_path)
    embed_and_save(args.target_language, laser, args.tgt_data_path, args.tgt_embed_path)

    src_embs = torch.load(args.src_embed_path)
    tgt_embs = torch.load(args.tgt_embed_path)
    print("Source embeddings: ", src_embs)
    print("Target embeddings: ", tgt_embs)
    
    src_langs = torch.tensor([[args.src_lang] for _ in range(len(src_embs))])
    tgt_langs = torch.tensor([[args.tgt_lang] for _ in range(len(tgt_embs))])
    print("Source language: ", src_langs)
    print("Target language: ", tgt_langs)

    train_src_emb, valid_src_emb, train_tgt_emb, valid_tgt_emb = train_valid_split(torch.tensor(src_embs), torch.tensor(tgt_embs), seed=args.seed)
    train_src_lang, valid_src_lang, train_tgt_lang, valid_tgt_lang = train_valid_split(torch.tensor(src_langs), torch.tensor(tgt_langs), seed=args.seed)

    print("Finished splitting")

    # Save as transformers.dataset format
    train = {"src_emb": train_src_emb, "tgt_emb": train_tgt_emb, "src_lang": train_src_lang, "tgt_lang": train_tgt_lang}
    torch.save(train, args.train_embed_path)
    print("Train embedding: ", train)
    print(len(train["src_emb"]))

    valid = {"src_emb": valid_src_emb, "tgt_emb": valid_tgt_emb, "src_lang": valid_src_lang, "tgt_lang": valid_tgt_lang,}
    torch.save(valid, args.valid_embed_path)
    print("Valid embedding: ", valid)
    print(len(valid["src_emb"]))
    print("\n==================================\n")


if __name__ == "__main__":
    main()
