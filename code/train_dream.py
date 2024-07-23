import argparse
import time
import torch
import yaml
import os
import importlib
from transformers import AutoModel
from utils import *


def train_pooler(model_loss, pooler, dataloader_train, dataloader_valid, optimizer, logging_path, save_pooler_path, train_type, device):
    min_valid_loss = float("inf")

    # Make logging_path
    log_dir = os.path.dirname(logging_path)
    os.makedirs(log_dir, exist_ok=True)

    start_time = time.time()

    with open(logging_path, 'w') as f:
        for epoch in range(10000):
            s_time = time.time()
            train_loss = 0
            meaning_embedding_losses = 0
            reconstruction_losses = 0
            language_embedding_losses = 0
            language_identification_losses = 0
            oracle_losses = 0
            
            for batch in dataloader_train:
                src_ids = batch["src_emb"].to(device)
                tgt_ids = batch["tgt_emb"].to(device)
                src_lang_label = batch["src_lang"].to(device)
                tgt_lang_label = batch["tgt_lang"].to(device)

                src_lang_label = src_lang_label.squeeze()
                src_lang_label = src_lang_label.to(torch.long)
                tgt_lang_label = tgt_lang_label.squeeze()
                tgt_lang_label = tgt_lang_label.to(torch.long)

                optimizer.zero_grad()
                loss, meaning_embedding_loss, reconstruction_loss, language_embedding_loss, language_identification_loss, oracle_loss = model_loss(src_ids, tgt_ids, src_lang_label, tgt_lang_label)
                
                train_loss += loss.item()
                if meaning_embedding_loss:
                    meaning_embedding_losses += meaning_embedding_loss.item()
                else:
                    meaning_embedding_losses += 0
                if reconstruction_loss:
                    reconstruction_losses += reconstruction_loss.item()
                else:
                    reconstruction_losses += 0
                if language_embedding_loss:
                    language_embedding_losses += language_embedding_loss.item()
                else:
                    language_embedding_losses += 0
                if language_identification_loss:
                    language_identification_losses += language_identification_loss.item()
                else:
                    language_identification_losses += 0
                if oracle_loss:
                    oracle_losses += oracle_loss.item()
                else:
                    oracle_losses += 0
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                valid_loss = 0
                for val_batch in dataloader_valid:
                    src_ids = val_batch["src_emb"].to(device)
                    tgt_ids = val_batch["tgt_emb"].to(device)
                    src_lang_label = val_batch["src_lang"].to(device)
                    tgt_lang_label = val_batch["tgt_lang"].to(device)

                    src_lang_label = src_lang_label.squeeze()
                    src_lang_label = src_lang_label.to(torch.long)
                    tgt_lang_label = tgt_lang_label.squeeze()
                    tgt_lang_label = tgt_lang_label.to(torch.long)

                    val_loss, _, _, _, _, _ = model_loss(
                        src_ids, tgt_ids, src_lang_label, tgt_lang_label
                    )
                    valid_loss += val_loss.item()

                # Remove calculating specific losses to speed up training
                status = (
                    f"Epoch:{epoch + 1: <2}, "
                    f"Train_loss: {train_loss / len(dataloader_train):.5f}, "
                    f"Meaning_embedding_loss: {meaning_embedding_losses / len(dataloader_train):.5f}, "
                    f"Reconstruction_loss: {reconstruction_losses / len(dataloader_train):.5f}, "
                    f"Language_embedding_loss: {language_embedding_losses / len(dataloader_train):.5f}, "
                    f"Language_identification_loss: {language_identification_losses / len(dataloader_train):.5f}, "
                    f"Oracle_loss: {oracle_losses / len(dataloader_train):.5f}, "
                    f"Valid_loss: {valid_loss / len(dataloader_valid):.5f}, "
                    f"{(time.time() - s_time) / 60:.1f}[min]"
                )
                print(status)
                f.write(status + "\n")

                if valid_loss < min_valid_loss:
                    epochs_no_improve = 0
                    min_valid_loss = valid_loss
                    torch.save(pooler.to("cpu").state_dict(), save_pooler_path)
                    pooler.to(device)
                else:
                    epochs_no_improve += 1

            if epochs_no_improve >= 10:
                break
    
    print("Total time: ",f"{(time.time() - start_time) / 60:.1f}[min]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file containing configurations")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # (1) Model, Pooler, Optimizer
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModel.from_pretrained(config["model_name_or_path"], cache_dir="/fs/clip-scratch/dayeonki/.cache").to(device)

    embedding_size = model.embeddings.word_embeddings.embedding_dim
    print("Embedding Size: ", embedding_size)
    PoolerClass = getattr(importlib.import_module("model.dream"), "MLP")
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=config["n_languages"])
    pooler.to(device)
    
    optimizer = torch.optim.Adam(pooler.parameters(), lr=float(config["learning_rate"]))

    # (2) Dataloader (embedded dataset)
    dataset_train, dataset_valid = load_dataset(config["train_path"], config["valid_path"])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=config["batch_size"], shuffle=False, drop_last=True)

    # (3) Loss class
    train_type_to_class = {"vanilla": "BaseLoss",
                           "oracle_old": "OracleLoss", 
                           "oracle_new": "OracleNewLoss", 
                           "oracle_orth": "OracleOrthLoss", 
                           "oracle_pair": "OraclePairLoss", 
                           "oracle_main": "OracleMainLoss", 
                           "oracle_all": "OracleAllLoss", 
                           "alone1": "Alone1Loss",
                           "alone2": "Alone2Loss", 
                           "alone3": "Alone3Loss",
                           "alone4": "Alone4Loss"}
    LossClass = getattr(importlib.import_module("loss.dream"), train_type_to_class[config["train_type"]])

    pooler_dir = os.path.dirname(config["save_pooler_path"])
    os.makedirs(pooler_dir, exist_ok=True)

    for weight in config["weights"]:
        print("Weight: ", weight)
        model_loss = LossClass(model, pooler, config["train_type"], weight, device)
        train_pooler(
            model_loss=model_loss,
            pooler=pooler, 
            dataloader_train=dataloader_train, 
            dataloader_valid=dataloader_valid,
            optimizer=optimizer, 
            logging_path=config["logging_path"],
            save_pooler_path=config["save_pooler_path"],
            train_type=config["train_type"], 
            device=device
        )


if __name__ == "__main__":
    main()
