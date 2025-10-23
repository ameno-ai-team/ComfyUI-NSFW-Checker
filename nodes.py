from torch.utils.data import Dataset
from transformers import pipeline
from PIL import Image
import numpy as np

class VideoFrameDataset(Dataset):
    """Dataset for video frames"""
    def __init__(self, images, interval):
        self.frames = []

        for index in range(len(images)):
            if interval != 0 and index % interval != 0:
                continue
            
            i = (images[index].cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(i)
            self.frames.append(img)
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return self.frames[idx]

class LoadNSFWTextModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["michellejieli/NSFW_text_classifier"],),
                "device": (["cuda", "cpu"],),
            },
        }
    
    RETURN_TYPES = ("TEXT_PIPELINE",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    
    def execute(self, model, device='cuda'):
        pipe = pipeline("text-classification", model=model, device=device,batch_size=32)
        return (pipe,)

class LoadNSFWVisionModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["AdamCodd/vit-base-nsfw-detector", "Falconsai/nsfw_image_detection"],),
                "device": (["cuda", "cpu"],),
            },
        }
    
    RETURN_TYPES = ("VISION_PIPELINE",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    
    def execute(self, model, device='cuda'):
        pipe = pipeline("image-classification", model=model, device=device,batch_size=32)
        return (pipe,)

class CheckNSFWText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TEXT_PIPELINE",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "FLOAT",)
    RETURN_NAMES = ("nsfw", "nsfw_score",)
    FUNCTION = "execute"
    
    def execute(self, model, text, threshold=0.5):
        result = model(text)
        
        if isinstance(result, list):
            result = result[0]
        
        if 'label' not in result or 'score' not in result:
            raise ValueError("Invalid result format from model. Expected 'label' and 'score' keys.")
        
        nsfw_score = result['score'] if result['label'] == 'NSFW' else 1 - result['score']
        avg_nsfw = round(nsfw_score, 4)
        return (avg_nsfw > threshold, avg_nsfw,)

class CheckNSFWVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VISION_PIPELINE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "interval": ("INT", {"default": 0, "min": 0, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "FLOAT",)
    RETURN_NAMES = ("nsfw", "nsfw_score",)
    FUNCTION = "execute"
    
    def execute(self, model, image, interval=0, threshold=0.5):
        dataset = VideoFrameDataset(image, interval)
        results = model(dataset.frames, batch_size=32)
        
        nsfw_scores = []
        for result in results:
            scores = {item["label"]: item["score"] for item in result}
            nsfw_scores.append(scores.get('nsfw', 0))
        
        avg_nsfw = round(np.mean(nsfw_scores), 4)
        return (avg_nsfw > threshold, avg_nsfw,)
    


NODE_CLASS_MAPPINGS = {
    "LoadNSFWTextModel": LoadNSFWTextModel,
    "LoadNSFWVisionModel": LoadNSFWVisionModel,
    "CheckNSFWVision": CheckNSFWVision,
    "CheckNSFWText": CheckNSFWText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadNSFWTextModel": "Load NSFW Text Model",
    "LoadNSFWVisionModel": "Load NSFW Vision Model",
    "CheckNSFWVision": "Check NSFW Vision",
    "CheckNSFWText": "Check NSFW Text",
}