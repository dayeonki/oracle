import torch
import os 
import torch.nn as nn
from dataclasses import dataclass
from utils_train import cosine_distance_torch
from torch.nn.functional import cross_entropy, mse_loss, cosine_embedding_loss as cos_emb_loss


@dataclass
class DREAMEmbedOutput:
    sentence_embed: torch.Tensor
    meaning_embed: torch.Tensor
    lang_embed: torch.Tensor
    lang_identification: torch.Tensor


class BaseLoss(nn.Module):
    def __init__(self, model, pooler, train_type, weights, device):
        super(BaseLoss, self).__init__()
        self.model = model
        self.pooler = pooler
        self.train_type = train_type
        self.weights = weights
        self.device = device
    
    def feed(self, input_ids):
        sentence_embed = self.model(input_ids).pooler_output
        meaning_embed, lang_embed, lang_identification = self.pooler(sentence_embed)
        return DREAMEmbedOutput(sentence_embed, meaning_embed, lang_embed, lang_identification)

    def train_feed(self, src_ids, tgt_ids):
        batch_size = src_ids.shape[0] // 2
        src_true_ids, src_random_ids = src_ids[:batch_size], src_ids[batch_size:]
        tgt_true_ids, tgt_random_ids = tgt_ids[:batch_size], tgt_ids[batch_size:]

        src_true_output = self.feed(src_true_ids)
        src_random_output = self.feed(src_random_ids)
        tgt_true_output = self.feed(tgt_true_ids)
        tgt_random_output = self.feed(tgt_random_ids)
        return src_true_output, src_random_output, tgt_true_output, tgt_random_output, batch_size

    def forward(self, src_ids, tgt_ids, src_lang_label, tgt_lang_label):
        src_true_output, src_random_output, tgt_true_output, tgt_random_output, batch_size = self.train_feed(src_ids, tgt_ids)

        y = torch.ones(batch_size).to(self.device)

        src_lang_label = torch.tensor(src_lang_label[:batch_size]).to(self.device)
        tgt_lang_label = torch.tensor(tgt_lang_label[:batch_size]).to(self.device)

        meaning_embedding_loss = (
            cos_emb_loss(src_true_output.meaning_embed, tgt_true_output.meaning_embed, y)
            + cos_emb_loss(src_true_output.meaning_embed, src_random_output.meaning_embed, -y)
            + cos_emb_loss(tgt_true_output.meaning_embed, tgt_random_output.meaning_embed, -y)
        )
        reconstruction_loss = (
            mse_loss(src_true_output.lang_embed + src_true_output.meaning_embed, src_true_output.sentence_embed) 
            + mse_loss(tgt_true_output.lang_embed + tgt_true_output.meaning_embed, tgt_true_output.sentence_embed)
        )
        language_embedding_loss = [
            cos_emb_loss(src_true_output.lang_embed, src_random_output.lang_embed, y), 
            cos_emb_loss(tgt_true_output.lang_embed, tgt_random_output.lang_embed, y),
            cos_emb_loss(src_true_output.lang_embed, tgt_true_output.lang_embed, -y),
        ]
        language_embedding_loss = language_embedding_loss[0] + language_embedding_loss[1]
        language_identification_loss = (
            cross_entropy(src_true_output.lang_identification, src_lang_label) 
            + cross_entropy(tgt_true_output.lang_identification, tgt_lang_label)
        )

        loss = self.weights[0] * meaning_embedding_loss + self.weights[1] * reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * language_identification_loss
        return loss, meaning_embedding_loss, reconstruction_loss, language_embedding_loss, language_identification_loss, None
    
    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        pooler_path = os.path.join(output_dir, "pooler_model.bin")
        torch.save(self.pooler.state_dict(), pooler_path)
        self.model.save_pretrained(output_dir)
                

class BOCLoss(BaseLoss):
    def forward(self, src_ids, tgt_ids, src_lang_label, tgt_lang_label):
        src_true_output, src_random_output, tgt_true_output, tgt_random_output, batch_size = self.train_feed(src_ids, tgt_ids)

        y = torch.ones(batch_size).to(self.device)
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        src_lang_label = torch.tensor(src_lang_label[:batch_size]).to(self.device)
        tgt_lang_label = torch.tensor(tgt_lang_label[:batch_size]).to(self.device)

        meaning_embedding_loss = (
            cos_emb_loss(src_true_output.meaning_embed, tgt_true_output.meaning_embed, y)
            + cos_emb_loss(src_true_output.meaning_embed, src_random_output.meaning_embed, -y)
            + cos_emb_loss(tgt_true_output.meaning_embed, tgt_random_output.meaning_embed, -y)
        )
        reconstruction_loss = (
            mse_loss(src_true_output.lang_embed + src_true_output.meaning_embed, src_true_output.sentence_embed) 
            + mse_loss(tgt_true_output.lang_embed + tgt_true_output.meaning_embed, tgt_true_output.sentence_embed)
        )
        language_embedding_loss = [
            cos_emb_loss(src_true_output.lang_embed, src_random_output.lang_embed, y), 
            cos_emb_loss(tgt_true_output.lang_embed, tgt_random_output.lang_embed, y),
            cos_emb_loss(src_true_output.lang_embed, tgt_true_output.lang_embed, -y),
        ]
        language_embedding_loss = language_embedding_loss[0] + language_embedding_loss[1]
        language_identification_loss = (
            cross_entropy(src_true_output.lang_identification, src_lang_label) 
            + cross_entropy(tgt_true_output.lang_identification, tgt_lang_label)
        )
    
        basic_orthogonal_loss = cos_similarity(src_true_output.lang_embed, tgt_true_output.lang_embed)
        basic_orthogonal_loss[basic_orthogonal_loss < 0] = 0  # hinge loss
        basic_orthogonal_loss = torch.mean(basic_orthogonal_loss)
        
        loss = self.weights[0] * meaning_embedding_loss + self.weights[1] * reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * language_identification_loss + self.weights[4] * basic_orthogonal_loss
        return loss, meaning_embedding_loss, reconstruction_loss, language_embedding_loss, language_identification_loss, basic_orthogonal_loss


class OPLLoss(BaseLoss):
    def forward(self, src_ids, tgt_ids, src_lang_label, tgt_lang_label):
        src_true_output, src_random_output, tgt_true_output, tgt_random_output, batch_size = self.train_feed(src_ids, tgt_ids)

        y = torch.ones(batch_size).to(self.device)
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        src_lang_label = torch.tensor(src_lang_label[:batch_size]).to(self.device)
        tgt_lang_label = torch.tensor(tgt_lang_label[:batch_size]).to(self.device)

        meaning_embedding_loss = (
            cos_emb_loss(src_true_output.meaning_embed, tgt_true_output.meaning_embed, y)
            + cos_emb_loss(src_true_output.meaning_embed, src_random_output.meaning_embed, -y)
            + cos_emb_loss(tgt_true_output.meaning_embed, tgt_random_output.meaning_embed, -y)
        )
        reconstruction_loss = (
            mse_loss(src_true_output.lang_embed + src_true_output.meaning_embed, src_true_output.sentence_embed) 
            + mse_loss(tgt_true_output.lang_embed + tgt_true_output.meaning_embed, tgt_true_output.sentence_embed)
        )
        language_embedding_loss = [
            cos_emb_loss(src_true_output.lang_embed, src_random_output.lang_embed, y), 
            cos_emb_loss(tgt_true_output.lang_embed, tgt_random_output.lang_embed, y),
            cos_emb_loss(src_true_output.lang_embed, tgt_true_output.lang_embed, -y),
        ]
        language_embedding_loss = language_embedding_loss[0] + language_embedding_loss[1]
        language_identification_loss = (
            cross_entropy(src_true_output.lang_identification, src_lang_label) 
            + cross_entropy(tgt_true_output.lang_identification, tgt_lang_label)
        )

        # Intra-class clustering
        intra_class_semantic = cos_similarity(src_true_output.meaning_embed, tgt_true_output.meaning_embed)
        intra_class_src = 1 - cosine_distance_torch(src_true_output.lang_embed)
        intra_class_tgt = 1 - cosine_distance_torch(tgt_true_output.lang_embed)
        intra_class_loss = torch.mean(3 - (intra_class_semantic + intra_class_src + intra_class_tgt))

        # Inter-class separation
        inter_class_src = abs(cos_similarity(src_true_output.meaning_embed, src_true_output.lang_embed))
        inter_class_tgt = abs(cos_similarity(tgt_true_output.meaning_embed, tgt_true_output.lang_embed))
        inter_class_lang = cos_similarity(src_true_output.lang_embed, tgt_true_output.lang_embed)
        inter_class_lang[inter_class_lang < 0] = 0  # hinge loss
        inter_class_loss = torch.mean(inter_class_src + inter_class_tgt + inter_class_lang)

        orthogonal_projection_loss = intra_class_loss + inter_class_loss
        loss = self.weights[0] * meaning_embedding_loss + self.weights[1] * reconstruction_loss + self.weights[2] * language_embedding_loss + self.weights[3] * language_identification_loss + self.weights[4] * orthogonal_projection_loss
        return loss, meaning_embedding_loss, reconstruction_loss, language_embedding_loss, language_identification_loss, orthogonal_projection_loss
