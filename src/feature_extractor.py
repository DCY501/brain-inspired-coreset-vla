"""
CLIP 多模态特征提取器
同时编码视觉观察与语言指令，输出对齐的多模态特征向量
"""
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTextModel
from config import CLIP_MODEL_NAME, VISUAL_FEATURE_DIM, TEXT_FEATURE_DIM


class CLIPFeatureExtractor(nn.Module):
    """
    基于 CLIP 的多模态特征提取器
    - 视觉分支: 冻结 CLIPVisionModel，输出 768 维视觉特征
    - 文本分支: 冻结 CLIPTextModel，输出 512 维文本特征
    - 单任务下文本特征为常数，但保留完整 512 维以与多任务 VLA 框架一致
    """
    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[CLIP] Loading model: {model_name}")
        
        # 加载 CLIP 视觉编码器（冻结）
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.vision_model.eval()
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        # 加载 CLIP 文本编码器（冻结）
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # 注意：单任务下文本特征为常数，但保留完整 512 维以与多任务 VLA 框架一致
        # 文本降维仅在多任务且需要轻量化的场景下考虑
        
        # 图像预处理工具
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.to(self.device)
        print(f"[CLIP] Loaded. Visual dim={VISUAL_FEATURE_DIM}, Text dim={TEXT_FEATURE_DIM}")
    
    @torch.no_grad()
    def extract_visual(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取视觉特征
        Args:
            images: (B, 3, H, W), float32, 值域 [0, 1] 或 [0, 255]
        Returns:
            visual_features: (B, 512), L2 归一化后的视觉特征
        """
        # CLIPProcessor 期望 PIL.Image 或 numpy，这里用 tensor 路径
        # 先确保值域正确并转到 device
        images = images.to(self.device)
        
        # 如果输入是 [0,1]，缩放到 [0,255] 并转为 uint8（processor 期望的格式）
        if images.max() <= 1.0:
            images = (images * 255).clamp(0, 255).byte()
        
        # processor 处理: resize, center_crop, normalize
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        pixel_values = inputs["pixel_values"].to(self.device)
        
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # pooler_output: (B, 512)
        visual_feats = vision_outputs.pooler_output
        # L2 归一化（CLIP 默认已归一化，这里再做一层保险）
        visual_feats = visual_feats / (visual_feats.norm(dim=1, keepdim=True) + 1e-8)
        return visual_feats.cpu()
    
    @torch.no_grad()
    def extract_text(self, texts: list) -> torch.Tensor:
        """
        提取文本特征
        Args:
            texts: list of str, 如 ["Pick up the cube..."] * B
        Returns:
            text_features: (B, 512), L2 归一化后的文本特征
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # pooler_output: (B, 512)
        text_feats = text_outputs.pooler_output
        
        # L2 归一化（CLIP 默认已归一化，保险起见再做一次）
        text_feats = text_feats / (text_feats.norm(dim=1, keepdim=True) + 1e-8)
        return text_feats.cpu()
    
    @torch.no_grad()
    def extract_multimodal(self, images: torch.Tensor, texts: list) -> tuple:
        """
        同时提取视觉和文本特征
        Args:
            images: (B, 3, H, W)
            texts: list of str
        Returns:
            visual_features: (B, 768)
            text_features: (B, 512)
        """
        v = self.extract_visual(images)
        t = self.extract_text(texts)
        return v, t
