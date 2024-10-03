import torch
from transformers import CLIPTokenizer, CLIPTextModel
from data.simulation.constants import movement_descriptions, easing_descriptions, angle_descriptions, shot_descriptions


def get_clip_embedding(texts, model_name="openai/clip-vit-large-patch14"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)

    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        text_features = text_encoder(**inputs).last_hidden_state.mean(dim=1)

    return text_features


def initialize_clip_embeddings(descriptions, model_name="openai/clip-vit-large-patch14"):
    clip_embeddings = {}
    for key, value in descriptions.items():
        clip_embeddings[key] = get_clip_embedding(value, model_name)
    return clip_embeddings


def initialize_all_clip_embeddings(clip_model_name="openai/clip-vit-large-patch14"):
    return {
        'movement': initialize_clip_embeddings(movement_descriptions, clip_model_name),
        'easing': initialize_clip_embeddings(easing_descriptions, clip_model_name),
        'angle': initialize_clip_embeddings(angle_descriptions, clip_model_name),
        'shot': initialize_clip_embeddings(shot_descriptions, clip_model_name)
    }
