# ControlNet控制整体布局，Stable Diffusion调整全局风格

# 导入所需的模块，用于图像处理、模型加载和生成图像
import torch  # PyTorch库，用于张量操作和模型推理
import numpy as np  # NumPy库，用于数组操作
from PIL import Image  # PIL库，用于图像加载和处理
import json  # JSON库，用于解析标签文件
import os  # OS库，用于文件路径操作
import torchvision.transforms as transforms  # Torchvision的变换模块，用于图像预处理
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation  # Transformers库，用于加载Mask2Former模型
from controlnet_aux import MLSDdetector  # ControlNet辅助工具，用于生成控制图像
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler  # Diffusers库，用于加载ControlNet和Stable Diffusion
from diffusers.utils import load_image  # Diffusers工具函数，用于加载图像
import matplotlib.pyplot as plt  # Matplotlib库，用于图像可视化

# 设置资源路径
RESOURCE_DIR = "resources"  # 资源根目录
MODELS_DIR = os.path.join(RESOURCE_DIR, "models")  # 模型目录
IMAGES_DIR = os.path.join(RESOURCE_DIR, "images")  # 图像目录
LABELS_DIR = os.path.join(RESOURCE_DIR, "labels")  # 标签目录
OUTPUT_DIR = os.path.join(RESOURCE_DIR, "output")  # 输出目录

# 设置全局变量
img_src = os.path.join(IMAGES_DIR, "sample_input.png")  # 输入图像的本地路径
# 如果资源目录中没有图片，则使用根目录中的图片
if not os.path.exists(img_src):
    img_src = "sample_input.png"
    
model_name = "runwayml/stable-diffusion-v1-5"  # Stable Diffusion模型的名称
mask_ID = 5  # 指定用于后续处理的掩码ID

# 从本地JSON文件加载ADE20K数据集的标签信息，用于语义分割的类别标注
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

# 定义函数，从语义分割图生成掩码
def get_mask_from_segmentation_map(seg_map: torch.Tensor):
    """从语义分割图生成掩码，每个掩码对应一个类别"""
    masks = []  # 存储生成的掩码列表
    labels = []  # 存储对应的类别标签列表
    for label in range(150):  # 遍历ADE20K的150个类别
        mask = np.ones((seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)  # 创建与分割图同尺寸的全1掩码
        indices = (seg_map == label)  # 找到分割图中等于当前类别ID的像素位置
        mask[indices] = 0  # 将对应类别的像素设为0，其他保持为1
        if indices.sum() > 0:  # 如果当前类别在图像中存在
            masks.append(mask)  # 将掩码添加到列表
            labels.append(label)  # 将类别ID添加到列表
    print(f"创建了 {len(masks)} 个掩码")  # 打印生成的掩码数量
    for idx, label in enumerate(labels):  # 遍历生成的掩码和标签
        print(f"索引: {idx}\t类别ID: {label}\t标签: {LABELS[str(label)]}")  # 打印每个掩码的索引、类别ID和标签名
    return masks, labels  # 返回掩码和标签列表

# 加载和预处理输入图像
image = load_image(img_src).resize((768, 512))  # 加载本地图像并调整大小为768x512像素

# 使用Mask2Former模型进行语义分割
print("=== 开始加载 Mask2Former 模型 ===")
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-ade-semantic", 
    cache_dir=MODELS_DIR
)  # 从本地加载预训练的图像处理器
inputs = processor(images=[image], return_tensors="pt")  # 预处理图像并转换为PyTorch张量
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-ade-semantic", 
    cache_dir=MODELS_DIR
)  # 从本地加载预训练的Mask2Former模型
print("=== Mask2Former 模型加载完成 ===")
print("=== 开始语义分割推理 ===")
outputs = model(**inputs)  # 运行模型进行语义分割推理
predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]  # 后处理获取语义分割图
print("=== 语义分割推理完成 ===")

# 从语义分割图生成掩码和对应的标签
masks, labels = get_mask_from_segmentation_map(predicted_semantic_map)  # 调用函数生成掩码和标签

# 加载MLSD检测器并生成控制图像，用于ControlNet输入
print("=== 开始加载 MLSD 检测器 ===")
processor = MLSDdetector.from_pretrained(
    "lllyasviel/Annotators", 
    cache_dir=MODELS_DIR
)  # 从本地加载预训练的MLSD检测器
print("=== MLSD 检测器加载完成 ===")
control_image = processor(image)  # 处理输入图像生成控制图像
checkpoint_name = "lllyasviel/control_v11p_sd15_mlsd"  # ControlNet检查点名称
model_name = "runwayml/stable-diffusion-v1-5"  # Stable Diffusion模型名称

# 加载ControlNet模型和Stable Diffusion生成管道
print("=== 开始加载 ControlNet 和 StableDiffusion 模型 ===")
controlnet = ControlNetModel.from_pretrained(
    checkpoint_name, 
    torch_dtype=torch.float16,  # 使用半精度浮点数以节省内存
    cache_dir=MODELS_DIR,  # 从本地加载模型
    use_safetensors=False  # 明确接受非safetensors格式
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_name, 
    controlnet=controlnet,  # 将ControlNet集成到管道中
    torch_dtype=torch.float16,  # 使用半精度浮点数
    cache_dir=MODELS_DIR,  # 从本地加载模型
    use_safetensors=False  # 明确接受非safetensors格式
)
print("=== ControlNet 和 StableDiffusion 模型加载完成 ===")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)  # 设置调度器为UniPCMultistepScheduler
pipe.enable_model_cpu_offload()  # 启用模型CPU卸载，优化GPU内存使用

# 准备掩码和控制图像
mask = torch.Tensor(masks[mask_ID]).repeat(3, 1, 1)  # 选取指定ID的掩码并重复为3通道，与控制图像匹配
object_mask = torch.Tensor(masks[mask_ID])  # 选取指定ID的掩码作为对象掩码
control = transforms.ToTensor()(control_image)  # 将控制图像转换为PyTorch张量
masked_control_image = transforms.ToPILImage()(mask * control)  # 使用掩码遮蔽控制图像并转换为PIL图像
object_mask = 1 - object_mask  # 反转掩码，0变为1，1变为0
object_mask = transforms.ToPILImage()(object_mask.unsqueeze(0))  # 将对象掩码转换为PIL图像

# 使用管道生成图像
prompt = ["A warm and cozy bedroom, furnished with blue and gray colors, and a wooden armchair. simplistic style"] * 4  # 定义生成图像的提示词，重复4次
generator = [torch.Generator(device="cuda").manual_seed(int(i)) for i in np.random.randint(50, size=len(prompt))]  # 为每张图像生成随机种子
print("=== 开始生成图像 ===")
print(f"当前内存使用情况：{torch.cuda.memory_allocated()/1024**2:.2f}MB")
output = pipe(
    prompt,  # 输入提示词
    image=masked_control_image,  # 输入遮蔽后的控制图像
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),  # 输入负面提示词，避免生成低质量图像
    num_inference_steps=30,  # 设置推理步数为30
    generator=generator,  # 使用指定的随机种子
)
print("=== 图像生成完成 ===")
print(f"生成后内存使用情况：{torch.cuda.memory_allocated()/1024**2:.2f}MB")

# 使用Matplotlib显示生成的图像网格
print("=== 开始图像可视化 ===")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 创建3x3的子图布局，设置画布大小为15x15英寸

# 显示原始图像
axes[0, 0].imshow(image)  # 在第1行第1列显示原始图像
axes[0, 0].set_title("Original Image")  # 设置标题
axes[0, 0].axis('off')  # 关闭坐标轴

# 显示遮蔽后的控制图像
axes[0, 1].imshow(masked_control_image)  # 在第1行第2列显示遮蔽后的控制图像
axes[0, 1].set_title("Masked Control Image")  # 设置标题
axes[0, 1].axis('off')  # 关闭坐标轴

# 显示对象掩码
axes[0, 2].imshow(object_mask, cmap='gray')  # 在第1行第3列显示对象掩码，使用灰度颜色映射
axes[0, 2].set_title("Object Mask")  # 设置标题
axes[0, 2].axis('off')  # 关闭坐标轴

# 显示四张生成的图像
axes[1, 0].imshow(output.images[0])  # 在第2行第1列显示第1张生成图像
axes[1, 0].set_title("Generated Image 1")  # 设置标题
axes[1, 0].axis('off')  # 关闭坐标轴

axes[1, 1].imshow(output.images[1])  # 在第2行第2列显示第2张生成图像
axes[1, 1].set_title("Generated Image 2")  # 设置标题
axes[1, 1].axis('off')  # 关闭坐标轴

axes[1, 2].imshow(output.images[2])  # 在第2行第3列显示第3张生成图像
axes[1, 2].set_title("Generated Image 3")  # 设置标题
axes[1, 2].axis('off')  # 关闭坐标轴

axes[2, 0].imshow(output.images[3])  # 在第3行第1列显示第4张生成图像
axes[2, 0].set_title("Generated Image 4")  # 设置标题
axes[2, 0].axis('off')  # 关闭坐标轴

# 关闭未使用的子图
axes[2, 1].axis('off')  # 关闭第3行第2列的子图
axes[2, 2].axis('off')  # 关闭第3行第3列的子图

plt.tight_layout()  # 调整子图布局以避免重叠
plt.show()  # 显示所有图像

# 保存生成的图像
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i, img in enumerate(output.images):
    img.save(os.path.join(OUTPUT_DIR, f"whole_image_{i+1}.png"))
print(f"生成的图像已保存到 {OUTPUT_DIR} 目录")