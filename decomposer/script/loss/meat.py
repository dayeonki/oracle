import torch
import os
import torch.nn as nn
from dataclasses import dataclass
from loss.meat import *
from utils_train import cosine_distance_torch


@dataclass
class MEATEmbedOutput:
    sentence_embed: torch.Tensor
    meaning_embed: torch.Tensor
    lang_embed: torch.Tensor


class DiscriminatorLoss(nn.Module):
    def __init__(self, model, pooler, discriminator):
        super(DiscriminatorLoss, self).__init__()
        self.model = model
        self.pooler = pooler
        self.discriminator = discriminator

    def forward(self, src_ids, tgt_ids, src_lang_label, tgt_lang_label):
        cross_fn = nn.CrossEntropyLoss()
        src_language = torch.squeeze(src_lang_label).long()
        tgt_language = torch.squeeze(tgt_lang_label).long()

        src_sentence_embed = self.model(src_ids).pooler_output
        tgt_sentence_embed = self.model(tgt_ids).pooler_output
        src_meaning_embed, src_lang_embed = self.pooler(src_sentence_embed)
        tgt_meaning_embed, tgt_lang_embed = self.pooler(tgt_sentence_embed)

        src_language_identification = self.discriminator(src_meaning_embed)
        tgt_language_identification = self.discriminator(tgt_meaning_embed)
        
        discriminator_loss = cross_fn(src_language_identification, src_language) + cross_fn(tgt_language_identification, tgt_language)
        return discriminator_loss


class BaseLoss(nn.Module):
    def __init__(self, model, pooler, discriminator, n_languages, train_type, weights, device):
        super(BaseLoss, self).__init__()
        self.model = model
        self.pooler = pooler
        self.discriminator = discriminator
        self.n_languages = n_languages
        self.train_type = train_type
        self.weights = weights
        self.device = device

    def feed(self, input_ids):
        sentence_embed = self.model(input_ids).pooler_output
        meaning_embed, lang_embed = self.pooler(sentence_embed)
        return MEATEmbedOutput(sentence_embed, meaning_embed, lang_embed)

    def train_feed(self, src_ids, tgt_ids):
        src_output = self.feed(src_ids)
        tgt_output = self.feed(tgt_ids)
        return src_output, tgt_output

    def forward(self, src_ids, tgt_ids, src_lang_label, tgt_lang_label):
        src_output, tgt_output = self.train_feed(src_ids, tgt_ids)

        cos_fn = nn.CosineEmbeddingLoss()
        cross_fn = nn.CrossEntropyLoss()
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        src_language_identification = self.discriminator(src_output.meaning_embed)
        tgt_language_identification = self.discriminator(tgt_output.meaning_embed)

        y = torch.ones(src_output.meaning_embed.size(0), device=self.device)
        y_adv = torch.full(src_language_identification.size(), fill_value=1 / self.n_languages, device=self.device)

        reconstruction_loss = (
            cos_fn(src_output.meaning_embed + src_output.lang_embed, src_output.sentence_embed, y) 
            + cos_fn(tgt_output.meaning_embed + tgt_output.lang_embed, tgt_output.sentence_embed, y)
        )
        cross_reconstruction_loss = (
            cos_fn(src_output.meaning_embed + tgt_output.lang_embed, tgt_output.sentence_embed, y) 
            + cos_fn(tgt_output.meaning_embed + src_output.lang_embed, src_output.sentence_embed, y)
        )
        language_embedding_loss = cos_fn(src_output.lang_embed, tgt_output.lang_embed, -y)
        adversarial_loss = (
            cross_fn(src_language_identification, y_adv) 
            + cross_fn(tgt_language_identification, y_adv)
        )

        # Add loss accordingly to train type
        if self.train_type == "vanilla":
            loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss
            return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, None
        
        elif self.train_type == "boc":
            basic_orthogonal_loss = cos_similarity(src_output.lang_embed, tgt_output.lang_embed)
            basic_orthogonal_loss[basic_orthogonal_loss < 0] = 0  # hinge loss
            basic_orthogonal_loss = torch.mean(basic_orthogonal_loss)
            loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * basic_orthogonal_loss
            return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, basic_orthogonal_loss
        
        elif self.train_type == "opl":
            # Intra-class clustering
            intra_class_semantic = cos_similarity(src_output.meaning_embed, tgt_output.meaning_embed)
            intra_class_src = 1 - cosine_distance_torch(src_output.lang_embed)
            intra_class_tgt = 1 - cosine_distance_torch(tgt_output.lang_embed)
            intra_class_loss = torch.mean(3 - (intra_class_semantic + intra_class_src + intra_class_tgt))

            # Inter-class separation
            inter_class_src = abs(cos_similarity(src_output.meaning_embed, src_output.lang_embed))
            inter_class_tgt = abs(cos_similarity(tgt_output.meaning_embed, tgt_output.lang_embed))
            inter_class_lang = cos_similarity(src_output.lang_embed, tgt_output.lang_embed)
            inter_class_lang[inter_class_lang < 0] = 0  # hinge loss
            inter_class_loss = torch.mean(inter_class_src + inter_class_tgt + inter_class_lang)

            orthogonal_projection_loss = intra_class_loss + inter_class_loss
            loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * orthogonal_projection_loss
            return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, orthogonal_projection_loss

    def save_pretrained(self, output_dir):
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
        pooler_path = os.path.join(output_dir, "pooler_model.bin")
        torch.save(self.pooler.state_dict(), pooler_path)
        self.model.save_pretrained(output_dir)


class BOCLoss(BaseLoss):
    def forward(self, src_ids, tgt_ids, src_lang_label, tgt_lang_label):
        src_output, tgt_output = self.train_feed(src_ids, tgt_ids)

        cos_fn = nn.CosineEmbeddingLoss()
        cross_fn = nn.CrossEntropyLoss()
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        src_language_identification = self.discriminator(src_output.meaning_embed)
        tgt_language_identification = self.discriminator(tgt_output.meaning_embed)

        y = torch.ones(src_output.meaning_embed.size(0), device=self.device)
        y_adv = torch.full(src_language_identification.size(), fill_value=1 / self.n_languages, device=self.device)

        reconstruction_loss = (
            cos_fn(src_output.meaning_embed + src_output.lang_embed, src_output.sentence_embed, y) 
            + cos_fn(tgt_output.meaning_embed + tgt_output.lang_embed, tgt_output.sentence_embed, y)
        )
        cross_reconstruction_loss = (
            cos_fn(src_output.meaning_embed + tgt_output.lang_embed, tgt_output.sentence_embed, y) 
            + cos_fn(tgt_output.meaning_embed + src_output.lang_embed, src_output.sentence_embed, y)
        )
        language_embedding_loss = cos_fn(src_output.lang_embed, tgt_output.lang_embed, -y)
        adversarial_loss = (
            cross_fn(src_language_identification, y_adv) 
            + cross_fn(tgt_language_identification, y_adv)
        )

        basic_orthogonal_loss = cos_similarity(src_output.lang_embed, tgt_output.lang_embed)
        basic_orthogonal_loss[basic_orthogonal_loss < 0] = 0  # hinge loss
        basic_orthogonal_loss = torch.mean(basic_orthogonal_loss)
        
        loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * basic_orthogonal_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, basic_orthogonal_loss
        

class OPLLoss(BaseLoss):
    def forward(self, src_ids, tgt_ids, src_lang_label, tgt_lang_label):
        src_output, tgt_output = self.train_feed(src_ids, tgt_ids)

        cos_fn = nn.CosineEmbeddingLoss()
        cross_fn = nn.CrossEntropyLoss()
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        src_language_identification = self.discriminator(src_output.meaning_embed)
        tgt_language_identification = self.discriminator(tgt_output.meaning_embed)

        y = torch.ones(src_output.meaning_embed.size(0), device=self.device)
        y_adv = torch.full(src_language_identification.size(), fill_value=1 / self.n_languages, device=self.device)

        reconstruction_loss = (
            cos_fn(src_output.meaning_embed + src_output.lang_embed, src_output.sentence_embed, y) 
            + cos_fn(tgt_output.meaning_embed + tgt_output.lang_embed, tgt_output.sentence_embed, y)
        )
        cross_reconstruction_loss = (
            cos_fn(src_output.meaning_embed + tgt_output.lang_embed, tgt_output.sentence_embed, y) 
            + cos_fn(tgt_output.meaning_embed + src_output.lang_embed, src_output.sentence_embed, y)
        )
        language_embedding_loss = cos_fn(src_output.lang_embed, tgt_output.lang_embed, -y)
        adversarial_loss = (
            cross_fn(src_language_identification, y_adv) 
            + cross_fn(tgt_language_identification, y_adv)
        )
    
        # Intra-class clustering
        intra_class_semantic = cos_similarity(src_output.meaning_embed, tgt_output.meaning_embed)
        intra_class_src = 1 - cosine_distance_torch(src_output.lang_embed)
        intra_class_tgt = 1 - cosine_distance_torch(tgt_output.lang_embed)
        intra_class_loss = torch.mean(3 - (intra_class_semantic + intra_class_src + intra_class_tgt))

        # Inter-class separation
        inter_class_src = abs(cos_similarity(src_output.meaning_embed, src_output.lang_embed))
        inter_class_tgt = abs(cos_similarity(tgt_output.meaning_embed, tgt_output.lang_embed))
        inter_class_lang = cos_similarity(src_output.lang_embed, tgt_output.lang_embed)
        inter_class_lang[inter_class_lang < 0] = 0  # hinge loss
        inter_class_loss = torch.mean(inter_class_src + inter_class_tgt + inter_class_lang)

        orthogonal_projection_loss = intra_class_loss + inter_class_loss
        loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * orthogonal_projection_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, orthogonal_projection_loss
