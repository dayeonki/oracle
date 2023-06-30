import argparse
import time
import torch
import yaml
import importlib
from utils import *
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from datasets import load_from_disk


def fine_tune_pooler(model_loss, discriminator_loss, dataloader_train, dataloader_valid, optimizer, discriminator_optimizer, accelerator, logging_path, save_pooler_path, train_type, batch_size):
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
                # Discriminator loss
                discriminator_optimizer.zero_grad()
                loss = discriminator_loss(**batch)
                accelerator.backward(loss)
                discriminator_optimizer.step()
                
                # Main loss
                optimizer.zero_grad()
                loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, additional_loss = model_loss(**batch)
                train_loss += loss.item()
                reconstruction_losses += reconstruction_loss.item()
                cross_reconstruction_losses += cross_reconstruction_loss.item()
                language_embedding_losses += language_embedding_loss.item()
                adversarial_losses += adversarial_loss.item()
                if train_type == "boc":
                    basic_orthogonal_losses += additional_loss.item()
                elif train_type == "opl":
                    orthogonal_projection_losses += additional_loss.item()
                accelerator.backward(loss)
                optimizer.step()

            with torch.no_grad():
                valid_loss = []
                for val_batch in dataloader_valid:
                    val_loss, _, _, _, _, _ = model_loss(**val_batch)
                    valid_loss.append(accelerator.gather_for_metrics(val_loss.repeat(batch_size)))
                valid_loss = torch.cat(valid_loss)
                valid_loss = torch.mean(valid_loss).item()

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
                accelerator.print(status)
                f.write(status + "\n")

                if valid_loss < min_valid_loss:
                    epochs_no_improve = 0
                    if accelerator.is_local_main_process:
                        unwrapped_model = accelerator.unwrap_model(model_loss)
                        unwrapped_model.save_pretrained(save_pooler_path)
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
    
    # (1) Set accelerator
    accelerator = Accelerator()

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # (2) Tokenizer, Model, Pooler
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    model = AutoModel.from_pretrained(config["model_name_or_path"])
    model.train()

    embedding_size = model.embeddings.word_embeddings.embedding_dim
    PoolerClass = getattr(importlib.import_module("model.meat"), "MLP")
    DiscriminatorClass = getattr(importlib.import_module("model.meat"), "Discriminator")
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=config["n_languages"])
    discriminator = DiscriminatorClass(embedding_size=embedding_size, n_languages=config["n_languages"])

    # (3) Loss Class
    train_type_to_class = {"vanilla": "BaseLoss", "boc": "BOCLoss", "opl": "OPLLoss"}
    LossClass = getattr(importlib.import_module("loss.meat"), train_type_to_class[config["train_type"]])
    DiscriminatorLossClass = getattr(importlib.import_module("loss.meat"), train_type_to_class[config["train_type"]])
    discriminator_loss = DiscriminatorLossClass(model=model, pooler=pooler, discriminator=discriminator)

    # (4) Dataloader (tokenized dataset)
    data_collator = LabeledBitextDataCollatorWithPadding(tokenizer, padding="longest", max_length=512)
    dataset_train = load_from_disk(config["train_path"])
    dataset_valid = load_from_disk(config["valid_path"])

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], collate_fn=data_collator, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=config["batch_size"], collate_fn=data_collator, drop_last=True)
    
    for weight in config["weights"]:
        accelerator.print(weight)
        model_loss = LossClass(model, pooler, discriminator, config["n_languages"], config["train_type"], weight, device)
        # Send both model and pooler parameters to optimizer
        optimizer = torch.optim.Adam(model_loss.parameters(), lr=float(config["learning_rate"]))
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=float(config["learning_rate"]))
        model_loss, optimizer, discriminator_optimizer, dataloader_train, dataloader_valid = accelerator.prepare(model_loss, optimizer, discriminator_optimizer, dataloader_train, dataloader_valid)
        fine_tune_pooler(
            model_loss=model_loss, 
            discriminator_loss=discriminator_loss, 
            dataloader_train=dataloader_train, 
            dataloader_valid=dataloader_valid, 
            optimizer=optimizer, 
            discriminator_optimizer=discriminator_optimizer, 
            accelerator=accelerator,
            logging_path=config["logging_path"], 
            save_pooler_path=config["save_pooler_path"], 
            train_type=config["train_type"], 
            batch_size=config["batch_size"]
        )


if __name__ == "__main__":
    main()
