import os
import json
import requests
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, CLIPProcessor, CLIPModel
from controlnet_aux import MLSDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline
import urllib.request
import shutil

# 创建资源目录
def create_directories():
    directories = [
        "resources",
        "resources/models",
        "resources/images",
        "resources/labels",
        "resources/output"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("目录结构创建完成")

# 下载ADE20K标签文件
def download_labels():
    url = "https://huggingface.co/datasets/huggingface/label-files/raw/main/ade20k-id2label.json"
    labels_path = "resources/labels/ade20k-id2label.json"
    response = requests.get(url)
    with open(labels_path, 'w') as f:
        f.write(response.text)
    print(f"标签文件已保存到: {labels_path}")

# 下载示例图片
def download_sample_image():
    raw_url = "https://raw.githubusercontent.com/naderAsadi/DesignGenie/main/examples/images/sample_input.png"
    img_path = "resources/images/sample_input.png"
    try:
        urllib.request.urlretrieve(raw_url, img_path)
        print(f"示例图片已保存到: {img_path}")
        # 同时拷贝到根目录，保持原脚本兼容
        shutil.copy(img_path, "sample_input.png")
    except Exception as e:
        print(f"图片下载失败: {e}")

# 下载模型文件
def download_models():
    print("正在下载模型，这可能需要一些时间...")
    
    # 1. 下载 Mask2Former 模型
    print("下载 Mask2Former 模型...")
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic", cache_dir="resources/models")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic", cache_dir="resources/models")
    print("Mask2Former 模型下载完成")
    
    # 2. 下载 MLSD 检测器
    print("下载 MLSD 检测器...")
    processor = MLSDdetector.from_pretrained("lllyasviel/Annotators", cache_dir="resources/models")
    print("MLSD 检测器下载完成")
    
    # 3. 下载 ControlNet 模型
    print("下载 ControlNet 模型...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_mlsd", 
        torch_dtype=torch.float16,
        cache_dir="resources/models",
        use_safetensors=False
    )
    print("ControlNet 模型下载完成")
    
    # 4. 下载 Stable Diffusion 模型
    print("下载 Stable Diffusion 模型...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir="resources/models",
        use_safetensors=False
    )
    print("Stable Diffusion 模型下载完成")
    
    # 5. 下载 Stable Diffusion Inpainting 模型 (用于 inpaint.py)
    print("下载 Stable Diffusion Inpainting 模型...")
    pipe_inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir="resources/models",
        use_safetensors=False
    )
    print("Stable Diffusion Inpainting 模型下载完成")
    
    # 6. 下载图像特征提取模型 (用于相似性搜索)
    print("下载图像特征提取模型...")
    try:
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            cache_dir="resources/models"
        )
        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", 
            cache_dir="resources/models"
        )
        print("图像特征提取模型下载完成")
    except Exception as e:
        print(f"图像特征提取模型下载失败: {e}")

if __name__ == "__main__":
    create_directories()
    download_labels()
    download_sample_image()
    download_models()
    print("所有资源下载完成！您可以将整个 'resources' 文件夹保存到本地使用。")