import argparse
import time
import torch
import yaml
import os
import importlib
from transformers import AutoModel
from utils import *


def train_pooler(model_loss, discriminator_loss, pooler, dataloader_train, dataloader_valid, optimizer, discriminator_optimizer, logging_path, save_pooler_path, train_type, device):
    min_valid_loss = float("inf")

    with open(logging_path, "w") as f:
        for epoch in range(10000):
            s_time = time.time()
            train_loss = 0
            reconstruction_losses = 0
            cross_reconstruction_losses = 0
            language_embedding_losses = 0
            adversarial_losses = 0
            basic_orthogonal_losses = 0
            orthogonal_projection_losses = 0
            
            for batch in dataloader_train:
                src_ids = batch["src_emb"].to(device)
                tgt_ids = batch["tgt_emb"].to(device)
                src_lang_label = batch["src_lang"].to(device)
                tgt_lang_label = batch["tgt_lang"].to(device)

                # Discriminator loss
                discriminator_optimizer.zero_grad()
                loss = discriminator_loss(src_ids, tgt_ids, src_lang_label, tgt_lang_label)
                loss.backward()
                discriminator_optimizer.step()

                # Main loss
                optimizer.zero_grad()
                loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, additional_loss = model_loss(src_ids, tgt_ids, src_lang_label, tgt_lang_label)
                train_loss += loss.item()
                reconstruction_losses += reconstruction_loss.item()
                cross_reconstruction_losses += cross_reconstruction_loss.item()
                language_embedding_losses += language_embedding_loss.item()
                adversarial_losses += adversarial_loss.item()
                if train_type == "boc":
                    basic_orthogonal_losses += additional_loss.item()
                elif train_type == "opl":
                    orthogonal_projection_losses += additional_loss.item()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                valid_loss = 0
                for val_batch in dataloader_valid:
                    src_ids = val_batch["src_emb"].to(device)
                    tgt_ids = val_batch["tgt_emb"].to(device)
                    src_lang_label = val_batch["src_lang"].to(device)
                    tgt_lang_label = val_batch["tgt_lang"].to(device)

                    val_loss, _, _, _, _, _ = model_loss(
                        src_ids, tgt_ids, src_lang_label, tgt_lang_label
                    )
                    valid_loss += val_loss.item()

                # Remove calculating specific losses to speed up training
                status = (
                    f"Epoch:{epoch + 1: <2}, "
                    f"Train_loss: {train_loss / len(dataloader_train):.5f}, "
                    f"Reconstruction_loss: {reconstruction_losses / len(dataloader_train):.5f}, "
                    f"Cross_reconstruction_loss: {cross_reconstruction_losses / len(dataloader_train):.5f}, "
                    f"Language_embedding_loss: {language_embedding_losses / len(dataloader_train):.5f}, "
                    f"Adversarial_loss: {adversarial_losses / len(dataloader_train):.5f}, "
                    f"Basic_orthogonal_loss: {basic_orthogonal_losses / len(dataloader_train):.5f}, "
                    f"Orthogonal_projection_loss: {orthogonal_projection_losses / len(dataloader_train):.5f}, "
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file containing configurations")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # (1) Model, Pooler, Optimizer
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(config["model_name_or_path"]).to(device)

    embedding_size = model.embeddings.word_embeddings.embedding_dim
    PoolerClass = getattr(importlib.import_module("model.meat"), "MLP")
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=config["n_languages"])
    pooler.to(device)
    DiscriminatorClass = getattr(importlib.import_module("model.meat"), "Discriminator")
    discriminator = DiscriminatorClass(embedding_size=embedding_size, n_languages=config["n_languages"])
    discriminator.to(device)
    
    optimizer = torch.optim.Adam(pooler.parameters(), lr=float(config["learning_rate"]))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=float(config["learning_rate"]))

    # (2) Dataloader (embedded dataset)
    dataset_train, dataset_valid = load_dataset(config["train_path"], config["valid_path"])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=config["batch_size"], shuffle=False)

    # (3) Loss class
    train_type_to_class = {"vanilla": "BaseLoss", "boc": "BOCLoss", "opl": "OPLLoss"}
    LossClass = getattr(importlib.import_module("loss.meat"), train_type_to_class[config["train_type"]])
    DiscriminatorLossClass = getattr(importlib.import_module("loss.meat"), train_type_to_class[config["train_type"]])
    discriminator_loss = DiscriminatorLossClass(model=model, pooler=pooler, discriminator=discriminator)

    os.makedirs(config["save_pooler_path"], exist_ok=True)
    
    for weight in config["weights"]:
        print(weight)
        model_loss = LossClass(model, pooler, discriminator, config["n_languages"], config["train_type"], weight, device)
        train_pooler(
            model_loss=model_loss,
            discriminator_loss=discriminator_loss,
            pooler=pooler, 
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid, 
            optimizer=optimizer, 
            discriminator_optimizer=discriminator_optimizer, 
            logging_path=config["logging_path"], 
            save_pooler_path=config["save_pooler_path"], 
            train_type=config["train_type"], 
            device=device
        )


if __name__ == "__main__":
    main()
