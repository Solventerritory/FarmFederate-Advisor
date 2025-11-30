# multimodal_model.py
"""
Multimodal model: RoBERTa (text) + ViT (image) + late-fusion head.
Text model uses PEFT/LoRA adapters for federated updates.
Image model: small classifier head; we keep base frozen (optionally fine-tunable).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, ViTModel, ViTConfig, ViTFeatureExtractor
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

TEXT_MODEL_NAME = "roberta-base"
IMAGE_MODEL_NAME = "google/vit-base-patch16-224-in21k"

NUM_LABELS = 5
ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]

class MultimodalClassifier(nn.Module):
    def __init__(self,
                 text_model_name=TEXT_MODEL_NAME,
                 image_model_name=IMAGE_MODEL_NAME,
                 text_embed_dim=768,
                 image_embed_dim=768,
                 fusion_hidden=512,
                 use_lora=True,
                 lora_r=8, lora_alpha=32, lora_dropout=0.05,
                 freeze_backbones=True):
        super().__init__()
        # Text backbone (Roberta)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        # create a simple pooling to get fixed vector
        self.text_pool = lambda outputs: outputs.last_hidden_state[:,0,:]  # rob_roberta[CLS] token pool

        # apply LoRA onto text backbone (target query/key/value / dense)
        self.use_lora = use_lora
        if use_lora:
            lcfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", task_type="SEQ_CLS"
            )
            # Wrap the AutoModel with PEFT (this will add adapter weights we can extract for FedAvg)
            self.text_backbone = get_peft_model(self.text_backbone, lcfg)

        # Image backbone (ViT)
        self.image_processor = ViTFeatureExtractor.from_pretrained(image_model_name)
        self.image_backbone = ViTModel.from_pretrained(image_model_name)

        # optionally freeze backbones
        if freeze_backbones:
            for p in self.text_backbone.parameters():
                p.requires_grad = False
            for p in self.image_backbone.parameters():
                p.requires_grad = False

        # projection heads (if necessary)
        self.text_proj = nn.Linear(text_embed_dim, fusion_hidden)
        self.image_proj = nn.Linear(image_embed_dim, fusion_hidden)

        # fusion + final classifier
        self.fusion = nn.Sequential(
            nn.Linear(fusion_hidden * 2, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, NUM_LABELS)
        )

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        # text path
        t_vec = None
        if input_ids is not None:
            t_out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
            t_pool = self.text_pool(t_out)
            t_vec = self.text_proj(t_pool)

        # image path
        i_vec = None
        if pixel_values is not None:
            i_out = self.image_backbone(pixel_values=pixel_values)
            # pool: take [CLS]
            i_pool = i_out.last_hidden_state[:, 0, :]
            i_vec = self.image_proj(i_pool)

        # if one modality missing, still allow predictions
        if t_vec is None:
            fused = torch.cat([torch.zeros_like(i_vec), i_vec], dim=1)
        elif i_vec is None:
            fused = torch.cat([t_vec, torch.zeros_like(t_vec)], dim=1)
        else:
            fused = torch.cat([t_vec, i_vec], dim=1)

        logits = self.fusion(fused)
        return logits

# helpers for adapter saving/loading
def get_text_adapter_state_dict(model: MultimodalClassifier):
    # returns peft state for text backbone if present
    try:
        sd = get_peft_model_state_dict(model.text_backbone)
        return sd
    except Exception:
        return {}

def set_text_adapter_state_dict(model: MultimodalClassifier, state_dict):
    try:
        set_peft_model_state_dict(model.text_backbone, state_dict)
    except Exception:
        # if not peft model, ignore
        pass

def get_image_head_state_dict(model: MultimodalClassifier):
    return {k:v.cpu() for k,v in model.image_proj.state_dict().items()}

def set_image_head_state_dict(model: MultimodalClassifier, state_dict):
    model.image_proj.load_state_dict(state_dict)
