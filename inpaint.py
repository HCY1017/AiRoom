# ControlNet控制整体布局，Stable Diffusion调整局部(inpaint)风格

# 导入必要的库
import torch  # PyTorch用于深度学习操作
import numpy as np  # 用于数值计算
from PIL import Image  # 图像处理
import json  # 用于解析JSON文件
import os  # 用于文件路径操作
import torchvision.transforms as transforms  # 图像转换工具
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation  # 用于语义分割
from controlnet_aux import MLSDdetector  # 线条检测器
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler  # 图像生成和修复
from diffusers.utils import load_image  # 图像加载工具
import matplotlib.pyplot as plt  # 可视化工具

# 设置资源路径
RESOURCE_DIR = "resources"  # 资源根目录
MODELS_DIR = os.path.join(RESOURCE_DIR, "models")  # 模型目录
IMAGES_DIR = os.path.join(RESOURCE_DIR, "images")  # 图像目录
LABELS_DIR = os.path.join(RESOURCE_DIR, "labels")  # 标签目录
OUTPUT_DIR = os.path.join(RESOURCE_DIR, "output")  # 输出目录

# 设置全局变量
img_src = os.path.join(IMAGES_DIR, "sample_input.png")  # 输入图像路径
# 如果资源目录中没有图片，则使用根目录中的图片
if not os.path.exists(img_src):
    img_src = "sample_input.png"
mask_ID = 4  # 掩码ID，用于选择要处理的区域

# 从本地JSON文件加载ADE20K数据集的标签信息，用于语义分割
labels_path = os.path.join(LABELS_DIR, "ade20k-id2label.json")
if os.path.exists(labels_path):
    with open(labels_path, 'r') as f:
        LABELS = json.load(f)
else:
    # 如果本地文件不存在，则从网络获取
    import requests
    print("本地标签文件不存在，从网络获取...")
    LABELS = requests.get("https://huggingface.co/datasets/huggingface/label-files/raw/main/ade20k-id2label.json").json()
    # 确保目录存在
    os.makedirs(LABELS_DIR, exist_ok=True)
    # 保存到本地
    with open(labels_path, 'w') as f:
        json.dump(LABELS, f)

def get_mask_from_segmentation_map(seg_map: torch.Tensor):
    """从分割图生成掩码，每个类别对应一个掩码"""
    masks, labels = [], []
    for label in range(150):  # ADE20K数据集有150个类别
        mask = np.ones((seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
        indices = (seg_map == label)
        mask[indices] = 0  # 将目标区域设为0，背景为1
        if indices.sum() > 0:  # 如果存在该类别
            masks.append(mask)
            labels.append(label)
    print(f"创建了 {len(masks)} 个掩码")
    for idx, label in enumerate(labels):
        print(f"索引: {idx}\t类别ID: {label}\t标签: {LABELS[str(label)]}")
    return masks, labels

# 加载和预处理输入图像
image = load_image(img_src).resize((768, 512))  # 调整图像大小为标准尺寸

# 使用Mask2Former进行语义分割
print("=== 开始加载 Mask2Former 模型 ===")
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-ade-semantic",
    cache_dir=MODELS_DIR
)
inputs = processor(images=[image], return_tensors="pt")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-ade-semantic",
    cache_dir=MODELS_DIR
)
print("=== Mask2Former 模型加载完成 ===")
print("=== 开始语义分割推理 ===")
outputs = model(**inputs)
predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
print("=== 语义分割推理完成 ===")

# 生成分割掩码
masks, labels = get_mask_from_segmentation_map(predicted_semantic_map)

# 使用MLSD检测器生成控制图像并与原始图像混合
print("=== 开始加载 MLSD 检测器 ===")
mlsd_processor = MLSDdetector.from_pretrained(
    "lllyasviel/Annotators", 
    cache_dir=MODELS_DIR
)
print("=== MLSD 检测器加载完成 ===")
control_image = mlsd_processor(image)  # 生成线条检测图
# 将控制图像和原始图像混合，创建更自然的控制引导
control_tensor = transforms.ToTensor()(control_image)
image_tensor = transforms.ToTensor()(image)
mixed_control_tensor = control_tensor * 0.5 + image_tensor * 0.5
mixed_control_image = transforms.ToPILImage()(mixed_control_tensor)

# 处理掩码并创建用于修复的遮罩图像
mask = torch.Tensor(masks[mask_ID])
# 生成修复用的掩码图像，0表示需要修复的区域
object_mask = 1 - mask
mask_image = transforms.ToPILImage()(object_mask.unsqueeze(0))

# 加载ControlNet和StableDiffusion修复模型
print("=== 开始加载 ControlNet 和 StableDiffusion Inpainting 模型 ===")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_mlsd",
    torch_dtype=torch.float16,
    cache_dir=MODELS_DIR,
    use_safetensors=False  # 明确接受非safetensors格式
)
# 创建带有ControlNet的图像修复管道
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    cache_dir=MODELS_DIR,
    use_safetensors=False  # 明确接受非safetensors格式
)
print("=== ControlNet 和 StableDiffusion Inpainting 模型加载完成 ===")
# 配置模型参数
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()  # 启用CPU卸载以节省显存

# 尝试启用xformers，如果安装了的话
try:
    pipe.enable_xformers_memory_efficient_attention()  # 尝试启用高效注意力机制
    print("成功启用 xformers 内存优化")
except (ModuleNotFoundError, ImportError):
    print("xformers 未安装，将使用默认注意力机制。如需更高效的内存使用，可以安装 xformers 库")

# 设置生成参数并生成图像
prompt = ["A luxurious Scandinavian style living room, minimalist furniture, natural wood elements, large windows with sunlight, cream colored walls, tasteful art pieces"] * 4
negative_prompt = ["cluttered, dark, oversaturated, poor quality, blurry, unrealistic, artificial lighting, overdecorated"] * 4
# 设置随机种子以确保可重复性
generator = [torch.Generator(device="cuda").manual_seed(int(i)) for i in np.random.randint(50, size=4)]

# 执行图像生成
print("=== 开始生成图像 ===")
print(f"当前内存使用情况：{torch.cuda.memory_allocated()/1024**2:.2f}MB")
output = pipe(
    prompt,
    image=image,  # 原始图像
    mask_image=mask_image,  # 指定需要修复的区域
    control_image=mixed_control_image,  # 控制图像用于引导生成
    negative_prompt=negative_prompt,
    num_inference_steps=30,  # 推理步数
    generator=generator,
    controlnet_conditioning_scale=0.7,  # 控制网络的影响程度
    guidance_scale=7.5,  # 提示词引导强度
)
print("=== 图像生成完成 ===")
print(f"生成后内存使用情况：{torch.cuda.memory_allocated()/1024**2:.2f}MB")

# 使用matplotlib显示结果
print("=== 开始图像可视化 ===")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# 显示原始图像和处理过程的中间结果
axes[0, 0].imshow(image)
axes[0, 0].set_title("Input Scene")
axes[0, 0].axis('off')

axes[0, 1].imshow(mixed_control_image)
axes[0, 1].set_title("Structure Guidance")
axes[0, 1].axis('off')

axes[0, 2].imshow(mask_image, cmap='gray')
axes[0, 2].set_title("Region to Redesign")
axes[0, 2].axis('off')

# 显示生成的图像
for i in range(4):
    row = (i // 3) + 1
    col = i % 3
    axes[row, col].imshow(output.images[i])
    axes[row, col].set_title(f"Design Variation {i+1}")
    axes[row, col].axis('off')

# 关闭未使用的子图
axes[2, 1].axis('off')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()

# 保存生成的图像
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i, img in enumerate(output.images):
    img.save(os.path.join(OUTPUT_DIR, f"inpaint_image_{i+1}.png"))
print(f"生成的图像已保存到 {OUTPUT_DIR} 目录")