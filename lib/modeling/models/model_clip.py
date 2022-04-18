import torch
import torch.nn as nn
from lib.modeling.models import clip


class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self, image):
        return self.model.encode_image(image)


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self, text):
        return self.model.encode_text(text)


class CLIP(nn.Module):
    def __init__(self, image_clip, text_clip) :
        super(CLIP, self).__init__()
        self.image_clip = image_clip
        self.text_clip = text_clip
        
    def forward(self, image, text):
        image_feat = self.image_clip(image)
        text_feat = self.text_clip(text)
        return image_feat, text_feat


def build_image_clip(model):
    return ImageCLIP(model)


def build_text_clip(model):
    return TextCLIP(model)


def build_clip(clip_model="ViT-B/32"):
    model, preprocess = clip.load(clip_model, device='cuda')
    image_clip = ImageCLIP(model)
    text_clip = TextCLIP(model)
    return CLIP(image_clip, text_clip)