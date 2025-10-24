from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# class LoadNSFWTextModel:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "model": (["michellejieli/NSFW_text_classifier"],),
#                 "device": (["cuda", "cpu"],),
#             },
#         }
    
#     RETURN_TYPES = ("TEXT_PIPELINE",)
#     RETURN_NAMES = ("model",)
#     FUNCTION = "execute"
    
#     def execute(self, model, device='cuda'):
#         pipe = pipeline("text-classification", model=model, device=device,batch_size=32)
#         return (pipe,)

class LoadNSFWVisionModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["AdamCodd/vit-base-nsfw-detector", "Falconsai/nsfw_image_detection"],),
                "device": (["cuda", "cpu"],),
            },
        }
    
    RETURN_TYPES = ("IMAGE_PROCESSOR", "IMAGE_CLASSIFICATION_MODEL",)
    RETURN_NAMES = ("processor", "model",)
    FUNCTION = "execute"
    
    def execute(self, model, device='cuda'):
        processor = AutoImageProcessor.from_pretrained(model, device=device, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(model).to(device)
        return (processor, model,)

# class CheckNSFWText:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "model": ("TEXT_PIPELINE",),
#                 "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
#             },
#             "optional": {
#                 "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
#             }
#         }
    
#     RETURN_TYPES = ("BOOLEAN", "FLOAT",)
#     RETURN_NAMES = ("nsfw", "nsfw_score",)
#     FUNCTION = "execute"
    
#     def execute(self, model, text, threshold=0.5):
#         result = model(text)
        
#         if isinstance(result, list):
#             result = result[0]
        
#         if 'label' not in result or 'score' not in result:
#             raise ValueError("Invalid result format from model. Expected 'label' and 'score' keys.")
        
#         nsfw_score = result['score'] if result['label'] == 'NSFW' else 1 - result['score']
#         avg_nsfw = round(nsfw_score, 4)
#         return (avg_nsfw > threshold, avg_nsfw,)

class CheckNSFWVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processor": ("IMAGE_PROCESSOR",),
                "model": ("IMAGE_CLASSIFICATION_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "interval": ("INT", {"default": 0, "min": 0, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("nsfw", "avg", "max", "min",)
    FUNCTION = "execute"
    
    def execute(self, processor, model, image, interval=0, threshold=0.5):
        frames = []
        for index in range(len(image)):
            if interval != 0 and index % interval != 0:
                continue
            
            i = (image[index].cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(i)
            frames.append(img)
            
        inputs = processor(frames, return_tensors="pt").to('cuda')
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)

        nsfw_label_id = None
        for label_id, label_name in model.config.id2label.items():
            if label_name == 'nsfw':
                nsfw_label_id = label_id
                break

        nsfw_scores_tensor = probs[:, nsfw_label_id]
        avg_nsfw = round(nsfw_scores_tensor.mean().item(), 4)
        max_nsfw = round(nsfw_scores_tensor.max().item(), 4)
        min_nsfw = round(nsfw_scores_tensor.min().item(), 4)
        
        return (avg_nsfw > threshold, avg_nsfw, max_nsfw, min_nsfw,)

NODE_CLASS_MAPPINGS = {
    # "LoadNSFWTextModel": LoadNSFWTextModel,
    "LoadNSFWVisionModel": LoadNSFWVisionModel,
    "CheckNSFWVision": CheckNSFWVision,
    # "CheckNSFWText": CheckNSFWText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "LoadNSFWTextModel": "Load NSFW Text Model",
    "LoadNSFWVisionModel": "Load NSFW Vision Model",
    "CheckNSFWVision": "Check NSFW Vision",
    # "CheckNSFWText": "Check NSFW Text",
}