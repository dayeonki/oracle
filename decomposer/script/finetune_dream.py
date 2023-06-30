import argparse
import time
import torch
import yaml
import importlib
from utils import *
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk


def fine_tune_pooler(model_loss, dataloader_train, dataloader_valid, optimizer, accelerator, logging_path, save_pooler_path, train_type, batch_size):
    min_valid_loss = float("inf")
    batch_size = batch_size // 2

    with open(logging_path, "w") as f:
        for epoch in range(1):
            s_time = time.time()
            train_loss = 0
            meaning_embedding_losses = 0
            reconstruction_losses = 0
            language_embedding_losses = 0
            language_identification_losses = 0
            basic_orthogonal_losses = 0
            orthogonal_projection_losses = 0
            
            for batch in dataloader_train:
                optimizer.zero_grad()
                loss, meaning_embedding_loss, reconstruction_loss, language_embedding_loss, language_identification_loss, additional_loss = model_loss(**batch)
                train_loss += loss.item()
                meaning_embedding_losses += meaning_embedding_loss.item()
                reconstruction_losses += reconstruction_loss.item()
                language_embedding_losses += language_embedding_loss.item()
                language_identification_losses += language_identification_loss.item()
                if train_type == "boc":
                    basic_orthogonal_losses += additional_loss.item()
                elif train_type == "opl":
                    orthogonal_projection_losses += additional_loss.item()
                accelerator.backward(loss)
                optimizer.step()

            with torch.no_grad():
                valid_loss = []
                # Distribute and gather all validation losses
                for val_batch in dataloader_valid:
                    val_loss, _, _, _, _, _ = model_loss(**val_batch)
                    valid_loss.append(accelerator.gather_for_metrics(val_loss.repeat(batch_size)))
                valid_loss = torch.cat(valid_loss)
                valid_loss = torch.mean(valid_loss).item()

                # Remove calculating specific losses to speed up training
                status = (
                    f"Epoch:{epoch + 1: <2}, "
                    f"Train_loss: {train_loss / len(dataloader_train):.5f}, "
                    f"Meaning_embedding_loss: {meaning_embedding_losses / len(dataloader_train):.5f}, "
                    f"Reconstruction_loss: {reconstruction_losses / len(dataloader_train):.5f}, "
                    f"Language_embedding_loss: {language_embedding_losses / len(dataloader_train):.5f}, "
                    f"Language_identification_loss: {language_identification_losses / len(dataloader_train):.5f}, "
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
    model.train()  # Set as train mode

    embedding_size = model.embeddings.word_embeddings.embedding_dim
    PoolerClass = getattr(importlib.import_module("model.dream"), "MLP")
    pooler = PoolerClass(embedding_size=embedding_size, n_languages=config["n_languages"])

    # (3) Loss class
    train_type_to_class = {"vanilla": "BaseLoss", "boc": "BOCLoss", "opl": "OPLLoss"}
    LossClass = getattr(importlib.import_module("loss.dream"), train_type_to_class[config["train_type"]])

    # (4) Dataloader (tokenized dataset)
    data_collator = LabeledBitextDataCollatorWithPadding(tokenizer, padding="longest", max_length=512)
    dataset_train = load_from_disk(config["train_path"])
    dataset_valid = load_from_disk(config["valid_path"])

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"], collate_fn=data_collator, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=config["batch_size"], collate_fn=data_collator, drop_last=True)

    for weight in config["weights"]:
        accelerator.print(weight)
        model_loss = LossClass(model, pooler, config["train_type"], weight, device)
        
        # Send both model and pooler parameters to optimizer
        optimizer = torch.optim.Adam(model_loss.parameters(), lr=float(config["learning_rate"]))
        model_loss, optimizer, dataloader_train, dataloader_valid = accelerator.prepare(model_loss, optimizer, dataloader_train, dataloader_valid)
        fine_tune_pooler(
            model_loss=model_loss, 
            dataloader_train=dataloader_train, 
            dataloader_valid=dataloader_valid, 
            optimizer=optimizer, 
            accelerator=accelerator,
            logging_path=config["logging_path"], 
            save_pooler_path=config["save_pooler_path"], 
            train_type=config["train_type"],
            batch_size=config["batch_size"]
        )


if __name__ == "__main__":
    main()
