import torch
import os
import statistics
import torch.nn as nn
from dataclasses import dataclass
from loss.meat import *


def cosine_distance_torch(x1, x2=None, eps=1e-6):
    x2 = x1 if x2 is None else x2
    w1 = x2.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x2 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


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
        src_sentence_embed = src_ids
        tgt_sentence_embed = tgt_ids
        src_meaning_embed, src_lang_embed = self.pooler(src_sentence_embed)
        tgt_meaning_embed, tgt_lang_embed = self.pooler(tgt_sentence_embed)

        src_language_identification = self.discriminator(src_meaning_embed)
        tgt_language_identification = self.discriminator(tgt_meaning_embed)
        
        discriminator_loss = cross_fn(src_language_identification, src_lang_label) + cross_fn(tgt_language_identification, tgt_lang_label)
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
        sentence_embed = input_ids
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

        loss = reconstruction_loss + cross_reconstruction_loss + language_embedding_loss + adversarial_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, None

    def save_pretrained(self, output_dir):
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
        pooler_path = os.path.join(output_dir, "pooler_model.bin")
        torch.save(self.pooler.state_dict(), pooler_path)
        self.model.save_pretrained(output_dir)

 

class OracleLoss(BaseLoss):
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

        oracle_loss = intra_class_loss + inter_class_loss
        loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * oracle_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, oracle_loss


class OracleAbsLoss(BaseLoss):
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
    
        # Pairwise distance
        intra_class_src = 1 - cosine_distance_torch(src_output.lang_embed)
        intra_class_tgt = 1 - cosine_distance_torch(tgt_output.lang_embed)
        intra_class_loss = torch.mean(2 - (intra_class_src + intra_class_tgt))

        # Orthogonality constraint
        inter_class_lang = cos_similarity(src_output.lang_embed, tgt_output.lang_embed)
        inter_class_lang = abs(inter_class_lang)
        inter_class_loss = torch.mean(inter_class_lang)

        oracle_loss = self.weights[0] * intra_class_loss + self.weights[1] * inter_class_loss
        loss = reconstruction_loss + cross_reconstruction_loss + language_embedding_loss + adversarial_loss + oracle_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, oracle_loss


class OracleNewLoss(BaseLoss):
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
    
        # Pairwise distance
        intra_class_src = 1 - cosine_distance_torch(src_output.lang_embed)
        intra_class_tgt = 1 - cosine_distance_torch(tgt_output.lang_embed)
        intra_class_loss = torch.mean(2 - (intra_class_src + intra_class_tgt))

        # Orthogonality constraint
        inter_class_lang = cos_similarity(src_output.lang_embed, tgt_output.lang_embed)
        inter_class_lang[inter_class_lang < 0] = 0  # hinge loss
        inter_class_loss = torch.mean(inter_class_lang)

        oracle_loss = self.weights[0] * intra_class_loss + self.weights[1] * inter_class_loss
        loss = reconstruction_loss + cross_reconstruction_loss + language_embedding_loss + adversarial_loss + oracle_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, oracle_loss



class OraclePairLoss(BaseLoss):
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
    
        # Pairwise distance
        intra_class_src = 1 - cosine_distance_torch(src_output.lang_embed) # src lang embed간의 similarity 작게
        intra_class_tgt = 1 - cosine_distance_torch(tgt_output.lang_embed) # tgt lang embed간의 similarity 작게
        oracle_loss = torch.mean(2 - (intra_class_src + intra_class_tgt))

        loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * oracle_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, oracle_loss


class OracleOrthLoss(BaseLoss):
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

        oracle_loss = cos_similarity(src_output.lang_embed, tgt_output.lang_embed)
        oracle_loss[oracle_loss < 0] = 0
        oracle_loss = torch.mean(oracle_loss)
        
        loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * oracle_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, oracle_loss


class OracleMainLoss(BaseLoss):
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

        # Orthogonalities
        oracle_loss2 = cos_similarity(src_output.lang_embed, src_output.meaning_embed)
        oracle_loss3 = cos_similarity(tgt_output.lang_embed, tgt_output.meaning_embed)
        oracle_loss2[oracle_loss2 < 0] = 0
        oracle_loss3[oracle_loss3 < 0] = 0
        oracle_loss2_mean = torch.mean(oracle_loss2)
        oracle_loss3_mean = torch.mean(oracle_loss3)
        mean_of_means_tensor = torch.stack([oracle_loss2_mean, oracle_loss3_mean])
        # Calculate the mean of means
        oracle_loss = torch.mean(mean_of_means_tensor)
        
        loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * oracle_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, oracle_loss


class OracleAllLoss(BaseLoss):
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

        # Orthogonalities
        oracle_loss1 = cos_similarity(src_output.lang_embed, tgt_output.lang_embed)
        oracle_loss2 = cos_similarity(src_output.lang_embed, src_output.meaning_embed)
        oracle_loss3 = cos_similarity(tgt_output.lang_embed, tgt_output.meaning_embed)
        oracle_loss1[oracle_loss1 < 0] = 0
        oracle_loss2[oracle_loss2 < 0] = 0
        oracle_loss3[oracle_loss3 < 0] = 0
        oracle_loss1_mean = torch.mean(oracle_loss1)
        oracle_loss2_mean = torch.mean(oracle_loss2)
        oracle_loss3_mean = torch.mean(oracle_loss3)
        mean_of_means_tensor = torch.stack([oracle_loss1_mean, oracle_loss2_mean, oracle_loss3_mean])
        # Calculate the mean of means
        oracle_loss = torch.mean(mean_of_means_tensor)
        
        loss = self.weights[0] * reconstruction_loss + self.weights[1] * cross_reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * adversarial_loss + self.weights[4] * oracle_loss
        return loss, reconstruction_loss, cross_reconstruction_loss, language_embedding_loss, adversarial_loss, oracle_loss


class Alone4Loss(BaseLoss):
    def forward(self, src_ids, tgt_ids, src_lang_label, tgt_lang_label):
        src_output, tgt_output = self.train_feed(src_ids, tgt_ids)
        cross_fn = nn.CrossEntropyLoss()
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        src_language_identification = self.discriminator(src_output.meaning_embed)
        tgt_language_identification = self.discriminator(tgt_output.meaning_embed)

        y_adv = torch.full(src_language_identification.size(), fill_value=1 / self.n_languages, device=self.device)

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

        oracle_loss = intra_class_loss + inter_class_loss
        loss = self.weights[3] * adversarial_loss + self.weights[4] * oracle_loss
        return loss, None, None, None, adversarial_loss, oracle_loss