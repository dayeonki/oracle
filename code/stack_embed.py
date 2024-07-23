import torch
import argparse


def list_of_strings(arg):
    return arg.split(',')


def cat(tensors, dim=0):
    return torch.cat(tuple(tensors), dim=dim)


def stack_multiple_embeddings(train_embed_list, valid_embed_list, train_embed_path, valid_embed_path):
    train_srcs = torch.empty(0)
    train_tgts = torch.empty(0)
    train_src_langs = torch.empty(0)
    train_tgt_langs = torch.empty(0)

    valid_srcs = torch.empty(0)
    valid_tgts = torch.empty(0)
    valid_src_langs = torch.empty(0)
    valid_tgt_langs = torch.empty(0)

    print("VALID")
    for valid_embed in valid_embed_list:
        v_embeddings = torch.load(valid_embed)
        valid_srcs = torch.cat((valid_srcs, v_embeddings["src_emb"]), dim=0)
        valid_tgts = torch.cat((valid_tgts, v_embeddings["tgt_emb"]), dim=0)
        valid_src_langs = torch.cat((valid_src_langs, v_embeddings["src_lang"]), dim=0)
        valid_tgt_langs = torch.cat((valid_tgt_langs, v_embeddings["tgt_lang"]), dim=0)
        print(valid_srcs.shape, valid_tgts.shape, valid_src_langs.shape, valid_tgt_langs.shape)
    
    valid = {"src_emb": valid_srcs, "tgt_emb": valid_tgts, "src_lang": valid_src_langs, "tgt_lang": valid_tgt_langs}
    torch.save(valid, valid_embed_path)
    
    print("\nTRAIN")
    for train_embed in train_embed_list:
        t_embeddings = torch.load(train_embed)
        train_srcs = torch.cat((train_srcs, t_embeddings["src_emb"]), dim=0)
        train_tgts = torch.cat((train_tgts, t_embeddings["tgt_emb"]), dim=0)
        train_src_langs = torch.cat((train_src_langs, t_embeddings["src_lang"]), dim=0)
        train_tgt_langs = torch.cat((train_tgt_langs, t_embeddings["tgt_lang"]), dim=0)
        print(train_srcs.shape, train_tgts.shape, train_src_langs.shape, train_tgt_langs.shape)
    
    train = {"src_emb": train_srcs, "tgt_emb": train_tgts, "src_lang": train_src_langs, "tgt_lang": train_tgt_langs}
    torch.save(train, train_embed_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embed_list", type=list_of_strings)
    parser.add_argument("--valid_embed_list", type=list_of_strings)
    parser.add_argument("--train_embed_path", type=str)
    parser.add_argument("--valid_embed_path", type=str)
    args = parser.parse_args()

    stack_multiple_embeddings(args.train_embed_list, args.valid_embed_list, args.train_embed_path, args.valid_embed_path)