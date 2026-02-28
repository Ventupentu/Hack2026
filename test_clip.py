import open_clip
import torch
clip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
print("Text enabled:", hasattr(clip_model, 'encode_text'))
