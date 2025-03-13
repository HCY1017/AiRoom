import os
import torch
import numpy as np
from PIL import Image
import json
import gradio as gr
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from controlnet_aux import MLSDdetector
from diffusers import (
    ControlNetModel, 
    StableDiffusionControlNetPipeline, 
    StableDiffusionControlNetInpaintPipeline, 
    UniPCMultistepScheduler
)
from diffusers.utils import load_image
import cv2

# 设置资源路径
RESOURCE_DIR = "resources"
MODELS_DIR = os.path.join(RESOURCE_DIR, "models")
IMAGES_DIR = os.path.join(RESOURCE_DIR, "images")
LABELS_DIR = os.path.join(RESOURCE_DIR, "labels")
OUTPUT_DIR = os.path.join(RESOURCE_DIR, "output")
GLOBAL_SAVE_DIR = os.path.join(OUTPUT_DIR, "global_style")  # 全局风格调整保存目录
LOCAL_SAVE_DIR = os.path.join(OUTPUT_DIR, "local_style")    # 局部风格调整保存目录

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GLOBAL_SAVE_DIR, exist_ok=True)
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

# 从本地JSON文件加载ADE20K数据集的标签信息
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

# 全局变量存储加载的模型
processor = None
mask2former_model = None
mlsd_processor = None
controlnet = None
global_pipe = None
inpaint_pipe = None
segmentation_result = None

def load_models():
    """加载所有需要的模型"""
    global processor, mask2former_model, mlsd_processor, controlnet, global_pipe, inpaint_pipe
    
    # 加载 Mask2Former 模型
    print("加载 Mask2Former 模型...")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-ade-semantic",
        cache_dir=MODELS_DIR
    )
    mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-ade-semantic",
        cache_dir=MODELS_DIR
    )
    
    # 加载 MLSD 检测器
    print("加载 MLSD 检测器...")
    mlsd_processor = MLSDdetector.from_pretrained(
        "lllyasviel/Annotators", 
        cache_dir=MODELS_DIR
    )
    
    # 加载 ControlNet 模型
    print("加载 ControlNet 模型...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_mlsd", 
        torch_dtype=torch.float16,
        cache_dir=MODELS_DIR,
        use_safetensors=False
    )
    
    # 加载全局风格调整管道
    print("加载 Stable Diffusion 全局风格调整模型...")
    global_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=MODELS_DIR,
        use_safetensors=False
    )
    global_pipe.scheduler = UniPCMultistepScheduler.from_config(global_pipe.scheduler.config)
    global_pipe.enable_model_cpu_offload()
    
    # 加载局部风格调整管道
    print("加载 Stable Diffusion Inpainting 局部风格调整模型...")
    inpaint_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=MODELS_DIR,
        use_safetensors=False
    )
    inpaint_pipe.scheduler = UniPCMultistepScheduler.from_config(inpaint_pipe.scheduler.config)
    inpaint_pipe.enable_model_cpu_offload()
    
    # 尝试启用xformers，如果安装了的话
    try:
        global_pipe.enable_xformers_memory_efficient_attention()
        inpaint_pipe.enable_xformers_memory_efficient_attention()
        print("成功启用 xformers 内存优化")
    except (ModuleNotFoundError, ImportError):
        print("xformers 未安装，将使用默认注意力机制")
    
    return "所有模型加载完成！"

def get_mask_from_segmentation_map(seg_map):
    """从分割图生成掩码，每个类别对应一个掩码"""
    masks, labels, label_names = [], [], []
    
    # 定义ADE20K标签的中文翻译
    chinese_labels = {
        "wall": "墙壁", "building": "建筑", "sky": "天空", "floor": "地板", "tree": "树",
        "ceiling": "天花板", "road": "道路", "bed": "床", "windowpane": "窗户", "grass": "草地",
        "cabinet": "柜子", "sidewalk": "人行道", "person": "人", "earth": "土地", "door": "门",
        "table": "桌子", "mountain": "山", "plant": "植物", "curtain": "窗帘", "chair": "椅子",
        "car": "汽车", "water": "水", "painting": "画", "sofa": "沙发", "shelf": "架子",
        "house": "房子", "sea": "海", "mirror": "镜子", "rug": "地毯", "field": "田野",
        "armchair": "扶手椅", "seat": "座位", "fence": "栅栏", "desk": "书桌", "rock": "岩石",
        "wardrobe": "衣柜", "lamp": "灯", "bathtub": "浴缸", "railing": "栏杆", "cushion": "靠垫",
        "base": "底座", "box": "盒子", "column": "柱子", "signboard": "招牌", "chest of drawers": "抽屉柜",
        "counter": "柜台", "sand": "沙子", "sink": "水槽", "skyscraper": "摩天大楼", "fireplace": "壁炉",
        "refrigerator": "冰箱", "grandstand": "看台", "path": "小路", "stairs": "楼梯", "runway": "跑道",
        "case": "箱子", "pool table": "台球桌", "pillow": "枕头", "screen door": "纱门", "stairway": "阶梯",
        "river": "河流", "bridge": "桥", "bookcase": "书柜", "blind": "百叶窗", "coffee table": "咖啡桌",
        "toilet": "马桶", "flower": "花", "book": "书", "hill": "山丘", "bench": "长凳",
        "countertop": "台面", "stove": "炉子", "palm": "棕榈树", "kitchen island": "厨房中岛", "computer": "电脑",
        "swivel chair": "旋转椅", "boat": "船", "bar": "吧台", "arcade machine": "街机", "hovel": "小屋",
        "bus": "公交车", "towel": "毛巾", "light": "灯光", "truck": "卡车", "tower": "塔",
        "chandelier": "吊灯", "awning": "遮阳篷", "streetlight": "路灯", "booth": "摊位", "television receiver": "电视机",
        "airplane": "飞机", "dirt track": "泥路", "apparel": "服装", "pole": "杆子", "land": "陆地",
        "bannister": "栏杆", "escalator": "自动扶梯", "ottoman": "脚凳", "bottle": "瓶子", "buffet": "自助餐",
        "poster": "海报", "stage": "舞台", "van": "货车", "ship": "轮船", "fountain": "喷泉",
        "conveyer belt": "传送带", "canopy": "天篷", "washer": "洗衣机", "plaything": "玩具", "swimming pool": "游泳池",
        "stool": "凳子", "barrel": "桶", "basket": "篮子", "waterfall": "瀑布", "tent": "帐篷",
        "bag": "包", "minibike": "小型摩托车", "cradle": "摇篮", "oven": "烤箱", "ball": "球",
        "food": "食物", "step": "台阶", "tank": "水箱", "trade name": "商标", "microwave": "微波炉",
        "pot": "锅", "animal": "动物", "bicycle": "自行车", "lake": "湖", "dishwasher": "洗碗机",
        "screen": "屏幕", "blanket": "毯子", "sculpture": "雕塑", "hood": "引擎盖", "sconce": "壁灯",
        "vase": "花瓶", "traffic light": "交通灯", "tray": "托盘", "ashcan": "垃圾桶", "fan": "风扇",
        "pier": "码头", "crt screen": "显示器", "plate": "盘子", "monitor": "显示器", "bulletin board": "公告板",
        "shower": "淋浴", "radiator": "暖气片", "glass": "玻璃", "clock": "时钟", "flag": "旗帜"
    }
    
    for label in range(150):  # ADE20K数据集有150个类别
        mask = np.ones((seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
        indices = (seg_map == label)
        mask[indices] = 0  # 将目标区域设为0，背景为1
        if indices.sum() > 0:  # 如果存在该类别
            masks.append(mask)
            labels.append(label)
            
            # 获取英文标签
            english_label = LABELS[str(label)]
            
            # 查找中文翻译，如果没有则使用英文
            chinese_label = chinese_labels.get(english_label, english_label)
            
            # 添加带有中文翻译的标签
            label_names.append(f"{label}: {english_label} - {chinese_label}")
    
    print(f"创建了 {len(masks)} 个掩码")
    for idx, label in enumerate(labels):
        print(f"索引: {idx}\t类别ID: {label}\t标签: {LABELS[str(label)]}")
    
    return masks, labels, label_names

def segment_image(image):
    """对图像进行语义分割"""
    global segmentation_result, processor, mask2former_model, mlsd_processor
    
    if processor is None or mask2former_model is None or mlsd_processor is None:
        return None, "请先加载模型！", []
    
    # 调整图像大小
    image_pil = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    image_pil = image_pil.resize((768, 512))
    
    # 进行语义分割
    inputs = processor(images=[image_pil], return_tensors="pt")
    outputs = mask2former_model(**inputs)
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image_pil.size[::-1]]
    )[0]
    
    # 生成分割掩码
    masks, labels, label_names = get_mask_from_segmentation_map(predicted_semantic_map)
    
    # 保存分割结果供后续使用
    segmentation_result = {
        "image": image_pil,
        "masks": masks,
        "labels": labels,
        "label_names": label_names,
        "semantic_map": predicted_semantic_map
    }
    
    # 生成控制图像
    control_image = mlsd_processor(image_pil)
    
    print(f"分割完成，找到 {len(label_names)} 个区域: {label_names}")
    
    return control_image, f"图像分割完成，找到 {len(label_names)} 个可调整区域", label_names

def adjust_global_style(prompt, negative_prompt, room_type, style_theme, num_steps, guidance_scale, num_images=4):
    """全局风格调整"""
    global segmentation_result, global_pipe, mlsd_processor
    
    if segmentation_result is None:
        return [None] * num_images + ["请先进行图像分割！"]
    
    if global_pipe is None or mlsd_processor is None:
        return [None] * num_images + ["请先加载模型！"]
    
    # 获取原始图像
    image = segmentation_result["image"]
    
    # 生成控制图像
    control_image = mlsd_processor(image)
    
    # 提取英文部分（去除中文描述）
    room_type = room_type.split(" - ")[0]
    style_theme = style_theme.split(" - ")[0]
    
    # 构建完整提示词，结合房间类型和风格主题
    full_prompt = f"A {style_theme} style {room_type}, {prompt}"
    
    # 设置生成参数
    prompts = [full_prompt] * num_images
    negative_prompts = [negative_prompt] * num_images
    generator = [torch.Generator(device="cuda").manual_seed(int(i)) for i in np.random.randint(1000, size=num_images)]
    
    # 执行图像生成
    output = global_pipe(
        prompts,
        image=control_image,  # 直接使用控制图像
        negative_prompt=negative_prompts,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=guidance_scale
    )
    
    # 保存生成的图像到临时位置
    for i, img in enumerate(output.images):
        img.save(os.path.join(OUTPUT_DIR, f"global_style_{i+1}.png"))
    
    # 返回单独的图像和状态文本，而不是列表+文本
    return output.images[0], output.images[1], output.images[2], output.images[3], "全局风格调整完成！"

def adjust_local_style(prompt, negative_prompt, mask_label, room_type, style_theme, num_steps, guidance_scale, num_images=4):
    """局部风格调整（Inpainting）"""
    global segmentation_result, inpaint_pipe, mlsd_processor
    
    if segmentation_result is None:
        return [None] * num_images + ["请先进行图像分割！"]
    
    if inpaint_pipe is None or mlsd_processor is None:
        return [None] * num_images + ["请先加载模型！"]
    
    # 获取原始图像和选定的掩码
    image = segmentation_result["image"]
    masks = segmentation_result["masks"]
    labels = segmentation_result["labels"]
    label_names = segmentation_result["label_names"]
    
    # 找到选定标签对应的掩码索引
    try:
        if mask_label is None or mask_label == "":
            return [None] * num_images + ["请选择要调整的区域"]
            
        # 找到选中的标签在label_names中的索引
        mask_id = label_names.index(mask_label)
    except (ValueError, IndexError, AttributeError):
        return [None] * num_images + ["无效的区域选择，请重新选择"]
    
    # 生成控制图像
    control_image = mlsd_processor(image)
    
    # 将控制图像和原始图像混合，创建更自然的控制引导
    control_tensor = transforms.ToTensor()(control_image)
    image_tensor = transforms.ToTensor()(image)
    mixed_control_tensor = control_tensor * 0.5 + image_tensor * 0.5
    mixed_control_image = transforms.ToPILImage()(mixed_control_tensor)
    
    # 处理掩码并创建用于修复的遮罩图像
    mask = torch.Tensor(masks[mask_id])
    object_mask = 1 - mask  # 反转掩码，0变为1，1变为0
    mask_image = transforms.ToPILImage()(object_mask.unsqueeze(0))
    
    # 提取英文部分（去除中文描述）
    room_type = room_type.split(" - ")[0]
    style_theme = style_theme.split(" - ")[0]
    
    # 构建完整提示词，结合房间类型和风格主题
    full_prompt = f"A {style_theme} style {room_type}, {prompt}"
    
    # 设置生成参数
    prompts = [full_prompt] * num_images
    negative_prompts = [negative_prompt] * num_images
    generator = [torch.Generator(device="cuda").manual_seed(int(i)) for i in np.random.randint(1000, size=num_images)]
    
    # 执行图像生成
    output = inpaint_pipe(
        prompts,
        image=image,
        mask_image=mask_image,
        control_image=mixed_control_image,
        negative_prompt=negative_prompts,
        num_inference_steps=num_steps,
        generator=generator,
        controlnet_conditioning_scale=0.7,
        guidance_scale=guidance_scale
    )
    
    # 保存生成的图像到临时位置
    for i, img in enumerate(output.images):
        img.save(os.path.join(OUTPUT_DIR, f"local_style_{i+1}.png"))
    
    # 返回单独的图像和状态文本，而不是列表+文本
    return output.images[0], output.images[1], output.images[2], output.images[3], "局部风格调整完成！"

# 显示选定区域的掩码
def display_selected_mask(mask_label):
    """根据选择的区域标签显示对应的掩码图像"""
    global segmentation_result
    
    if segmentation_result is None:
        return None, "请先进行图像分割！"
    
    if mask_label is None or mask_label == "":
        return None, "请选择要调整的区域"
    
    try:
        # 获取掩码和标签
        masks = segmentation_result["masks"]
        label_names = segmentation_result["label_names"]
        image = segmentation_result["image"]
        
        # 找到选中的标签在label_names中的索引
        mask_id = label_names.index(mask_label)
        
        # 获取对应的掩码
        mask = masks[mask_id]
        
        # 创建彩色掩码图像以便更好地可视化
        # 创建RGB图像，将选中区域标记为红色
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_rgb[mask == 0] = [255, 0, 0]  # 红色表示选中的区域
        
        # 将原始图像和掩码混合，使掩码半透明
        image_np = np.array(image)
        image_np = cv2.resize(image_np, (mask.shape[1], mask.shape[0]))
        
        # 创建混合图像
        alpha = 0.5
        mask_overlay = cv2.addWeighted(image_np, 1 - alpha, mask_rgb, alpha, 0)
        
        # 将NumPy数组转换为PIL图像
        mask_image = Image.fromarray(mask_overlay)
        
        return mask_image, f"已选择区域: {mask_label}"
    except (ValueError, IndexError, AttributeError) as e:
        print(f"显示掩码时出错: {e}")
        return None, f"无法显示所选区域: {str(e)}"

# 保存设计方案
def save_global_style(image_indices, room_type, style_theme):
    """保存全局风格调整的设计方案"""
    if not image_indices:
        return "请至少选择一个设计方案进行保存"
    
    # 提取英文部分（去除中文描述）
    room_type_en = room_type.split(" - ")[0]
    style_theme_en = style_theme.split(" - ")[0]
    
    # 生成基础文件名
    base_filename = f"{room_type_en}_{style_theme_en}"
    
    saved_paths = []
    for idx in image_indices:
        # 查找已有的同类型文件，确定新文件的编号
        existing_files = [f for f in os.listdir(GLOBAL_SAVE_DIR) if f.startswith(base_filename)]
        file_num = len(existing_files) + 1
        
        # 创建最终文件名
        filename = f"{base_filename}_{file_num}.png"
        save_path = os.path.join(GLOBAL_SAVE_DIR, filename)
        
        # 复制临时文件到保存目录
        temp_file = os.path.join(OUTPUT_DIR, f"global_style_{idx}.png")
        if os.path.exists(temp_file):
            try:
                # 使用PIL打开并保存图像，确保格式正确
                img = Image.open(temp_file)
                img.save(save_path)
                saved_paths.append(save_path)
            except Exception as e:
                return f"保存方案 {idx} 失败: {str(e)}"
        else:
            return f"找不到方案 {idx} 的图像，请先生成设计方案"
    
    if len(saved_paths) == 1:
        return f"已保存设计方案到 {saved_paths[0]}"
    else:
        return f"已成功保存 {len(saved_paths)} 个设计方案"

def save_local_style(image_indices, room_type, style_theme, mask_label):
    """保存局部风格调整的设计方案"""
    if not image_indices:
        return "请至少选择一个设计方案进行保存"
    
    # 提取英文部分（去除中文描述）
    room_type_en = room_type.split(" - ")[0]
    style_theme_en = style_theme.split(" - ")[0]
    
    # 从mask_label中提取区域信息
    area_info = mask_label.split(":")[0].strip() if mask_label and ":" in mask_label else "area"
    
    # 生成基础文件名
    base_filename = f"{room_type_en}_{style_theme_en}_area{area_info}"
    
    saved_paths = []
    for idx in image_indices:
        # 查找已有的同类型文件，确定新文件的编号
        existing_files = [f for f in os.listdir(LOCAL_SAVE_DIR) if f.startswith(base_filename)]
        file_num = len(existing_files) + 1
        
        # 创建最终文件名
        filename = f"{base_filename}_{file_num}.png"
        save_path = os.path.join(LOCAL_SAVE_DIR, filename)
        
        # 复制临时文件到保存目录
        temp_file = os.path.join(OUTPUT_DIR, f"local_style_{idx}.png")
        if os.path.exists(temp_file):
            try:
                # 使用PIL打开并保存图像，确保格式正确
                img = Image.open(temp_file)
                img.save(save_path)
                saved_paths.append(save_path)
            except Exception as e:
                return f"保存方案 {idx} 失败: {str(e)}"
        else:
            return f"找不到方案 {idx} 的图像，请先生成设计方案"
    
    if len(saved_paths) == 1:
        return f"已保存设计方案到 {saved_paths[0]}"
    else:
        return f"已成功保存 {len(saved_paths)} 个设计方案"

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="AI房间设计助手", css="""
        #region-dropdown .wrap {
            max-height: 300px;
            overflow-y: auto;
            z-index: 999;
            position: relative;
        }
        #region-dropdown .wrap::-webkit-scrollbar {
            width: 10px;
        }
        #region-dropdown .wrap::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        #region-dropdown .wrap::-webkit-scrollbar-thumb {
            background: #888;
        }
        #region-dropdown .wrap::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    """) as app:
        gr.Markdown("# AI房间设计助手")
        gr.Markdown("## 使用ControlNet和Stable Diffusion进行房间风格调整")
        
        # 定义房间类型和风格主题选项
        room_types = [
            "living room - 客厅", 
            "bedroom - 卧室", 
            "kitchen - 厨房", 
            "bathroom - 浴室", 
            "dining room - 餐厅", 
            "office - 办公室", 
            "study room - 书房", 
            "children's room - 儿童房"
        ]
        
        style_themes = [
            "modern - 现代", 
            "minimalist - 极简", 
            "Scandinavian - 北欧", 
            "industrial - 工业风", 
            "rustic - 乡村", 
            "traditional - 传统", 
            "contemporary - 当代", 
            "mid-century modern - 中世纪现代", 
            "bohemian - 波西米亚", 
            "coastal - 海岸风", 
            "farmhouse - 农舍", 
            "luxury - 奢华"
        ]
        
        # 定义提示词预设
        prompt_presets = {
            "简约舒适": "clean lines, comfortable seating, natural light, warm tones, simple decor",
            "奢华典雅": "elegant furnishings, crystal chandelier, marble surfaces, plush seating, gold accents",
            "自然原木": "wooden furniture, plants, natural materials, earth tones, organic textures",
            "明亮通透": "large windows, white walls, light wood floors, minimal furniture, airy space",
            "复古怀旧": "vintage furniture, retro color palette, antique accessories, classic patterns",
            "工业风格": "exposed brick, metal fixtures, concrete floors, raw materials, minimal decor",
            "温馨家庭": "comfortable seating, soft textiles, family photos, warm lighting, cozy atmosphere",
            "艺术创意": "colorful accents, unique art pieces, creative lighting, bold patterns, artistic elements"
        }
        
        # 定义负面提示词预设
        negative_prompt_presets = {
            "标准负面提示词": "cluttered, dark, oversaturated, poor quality, blurry, unrealistic",
            "避免过度装饰": "over decorated, cluttered, busy, chaotic, messy, disorganized",
            "避免昏暗效果": "dark, gloomy, dim, shadowy, poorly lit, murky",
            "避免不真实效果": "unrealistic, cartoon, anime, illustration, painting, drawing, 3d render",
            "避免低质量": "poor quality, low resolution, blurry, noisy, distorted, deformed",
            "避免人物": "people, person, human, face, hands, fingers",
            "避免文字": "text, letters, words, signage, labels, logos",
            "避免奇怪构图": "cropped, cut off, weird angle, distorted perspective, bad composition"
        }
        
        # 模型加载按钮
        with gr.Row():
            load_models_btn = gr.Button("加载模型")
            model_status = gr.Textbox(label="模型状态", value="未加载")
        
        # 创建选项卡界面
        with gr.Tabs() as tabs:
            # 全局风格调整选项卡
            with gr.TabItem("全局风格调整"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 输入区域
                        input_image = gr.Image(label="输入图像", type="pil")
                        segment_btn = gr.Button("分析图像结构")
                        
                        # 参数设置
                        room_type = gr.Dropdown(label="房间类型", choices=room_types, value="living room - 客厅")
                        style_theme = gr.Dropdown(label="主题风格", choices=style_themes, value="modern - 现代")
                        
                        # 提示词预设和输入
                        prompt_preset = gr.Dropdown(label="提示词预设", choices=list(prompt_presets.keys()), value="简约舒适")
                        prompt = gr.Textbox(label="提示词", value=prompt_presets["简约舒适"])
                        
                        # 负面提示词预设和输入
                        negative_prompt_preset = gr.Dropdown(label="负面提示词预设", choices=list(negative_prompt_presets.keys()), value="标准负面提示词")
                        negative_prompt = gr.Textbox(label="负面提示词", value=negative_prompt_presets["标准负面提示词"])
                        
                        num_steps = gr.Slider(label="推理步数", minimum=10, maximum=50, step=1, value=30)
                        guidance_scale = gr.Slider(label="引导比例", minimum=1.0, maximum=15.0, step=0.1, value=7.5)
                        
                        # 生成按钮
                        generate_btn = gr.Button("生成设计方案")
                    
                    with gr.Column(scale=1):
                        # 预览区域
                        control_image = gr.Image(label="结构控制图像")
                        status_text = gr.Textbox(label="状态信息")
                        
                        # 结果展示区域
                        gr.Markdown("### 设计方案")
                        with gr.Row():
                            output_images = [gr.Image(label=f"方案 {i+1}") for i in range(2)]
                        with gr.Row():
                            output_images.extend([gr.Image(label=f"方案 {i+3}") for i in range(2)])
                        
                        # 保存按钮区域
                        gr.Markdown("### 保存设计方案")
                        with gr.Row():
                            save_image_index = gr.CheckboxGroup(label="选择要保存的方案", choices=["方案 1", "方案 2", "方案 3", "方案 4"], value=[])
                            save_btn = gr.Button("保存选中的设计方案")
                        save_status = gr.Textbox(label="保存状态")
            
            # 局部风格调整选项卡
            with gr.TabItem("局部风格调整"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 输入区域
                        input_image_local = gr.Image(label="输入图像", type="pil")
                        segment_btn_local = gr.Button("分析图像结构")
                        
                        # 参数设置
                        region_choices = gr.Textbox(visible=False)  # 隐藏的文本框用于存储区域选项
                        with gr.Row(elem_id="region-dropdown"):
                            mask_label_local = gr.Dropdown(label="选择调整区域", choices=[], interactive=True)
                        room_type_local = gr.Dropdown(label="房间类型", choices=room_types, value="living room - 客厅")
                        style_theme_local = gr.Dropdown(label="主题风格", choices=style_themes, value="modern - 现代")
                        
                        # 提示词预设和输入
                        prompt_preset_local = gr.Dropdown(label="提示词预设", choices=list(prompt_presets.keys()), value="简约舒适")
                        prompt_local = gr.Textbox(label="提示词", value=prompt_presets["简约舒适"])
                        
                        # 负面提示词预设和输入
                        negative_prompt_preset_local = gr.Dropdown(label="负面提示词预设", choices=list(negative_prompt_presets.keys()), value="标准负面提示词")
                        negative_prompt_local = gr.Textbox(label="负面提示词", value=negative_prompt_presets["标准负面提示词"])
                        
                        num_steps_local = gr.Slider(label="推理步数", minimum=10, maximum=50, step=1, value=30)
                        guidance_scale_local = gr.Slider(label="引导比例", minimum=1.0, maximum=15.0, step=0.1, value=7.5)
                        
                        # 生成按钮
                        generate_btn_local = gr.Button("生成设计方案")
                        update_regions_btn = gr.Button("更新区域列表", visible=False)  # 隐藏的按钮用于触发更新
                    
                    with gr.Column(scale=1):
                        # 预览区域
                        control_image_local = gr.Image(label="区域掩码图像")
                        status_text_local = gr.Textbox(label="状态信息")
                        
                        # 结果展示区域
                        gr.Markdown("### 设计方案")
                        with gr.Row():
                            output_images_local = [gr.Image(label=f"方案 {i+1}") for i in range(2)]
                        with gr.Row():
                            output_images_local.extend([gr.Image(label=f"方案 {i+3}") for i in range(2)])
                        
                        # 保存按钮区域
                        gr.Markdown("### 保存设计方案")
                        with gr.Row():
                            save_image_index_local = gr.CheckboxGroup(label="选择要保存的方案", choices=["方案 1", "方案 2", "方案 3", "方案 4"], value=[])
                            save_btn_local = gr.Button("保存选中的设计方案")
                        save_status_local = gr.Textbox(label="保存状态")
        
        # 设置事件处理
        load_models_btn.click(load_models, inputs=[], outputs=[model_status])
        
        # 全局风格调整事件
        segment_btn.click(
            segment_image, 
            inputs=[input_image], 
            outputs=[control_image, status_text, region_choices]
        )
        
        # 提示词预设选择事件
        def update_prompt(preset_name):
            return prompt_presets.get(preset_name, "")
            
        def update_negative_prompt(preset_name):
            return negative_prompt_presets.get(preset_name, "")
            
        prompt_preset.change(
            update_prompt,
            inputs=[prompt_preset],
            outputs=[prompt]
        )
        
        negative_prompt_preset.change(
            update_negative_prompt,
            inputs=[negative_prompt_preset],
            outputs=[negative_prompt]
        )
        
        # 局部风格调整的提示词预设选择事件
        prompt_preset_local.change(
            update_prompt,
            inputs=[prompt_preset_local],
            outputs=[prompt_local]
        )
        
        negative_prompt_preset_local.change(
            update_negative_prompt,
            inputs=[negative_prompt_preset_local],
            outputs=[negative_prompt_local]
        )
        
        generate_btn.click(
            adjust_global_style, 
            inputs=[prompt, negative_prompt, room_type, style_theme, num_steps, guidance_scale], 
            outputs=output_images + [status_text]
        )
        
        # 局部风格调整事件
        # 分割图像并存储区域列表
        def process_segmentation_local(image):
            control_img, status, label_choices = segment_image(image)
            # 将选项列表转换为字符串存储
            choices_str = "|||".join(label_choices)
            return control_img, status, choices_str
            
        # 更新下拉菜单选项
        def update_dropdown(choices_str):
            if not choices_str:
                return gr.Dropdown(choices=[])
            choices = choices_str.split("|||")
            return gr.Dropdown(choices=choices)
            
        segment_btn_local.click(
            process_segmentation_local, 
            inputs=[input_image_local], 
            outputs=[control_image_local, status_text_local, region_choices]
        )
        
        # 使用region_choices更新下拉菜单
        region_choices.change(
            update_dropdown,
            inputs=[region_choices],
            outputs=[mask_label_local]
        )
        
        # 当用户选择区域时，更新掩码图像
        mask_label_local.change(
            display_selected_mask,
            inputs=[mask_label_local],
            outputs=[control_image_local, status_text_local]
        )
        
        generate_btn_local.click(
            adjust_local_style, 
            inputs=[prompt_local, negative_prompt_local, mask_label_local, room_type_local, style_theme_local, num_steps_local, guidance_scale_local], 
            outputs=output_images_local + [status_text_local]
        )
        
        # 保存设计方案事件
        def process_save_global(image_indices, room_type, style_theme):
            # 从选择的方案中提取索引号
            indices = [int(idx.split(" ")[1]) for idx in image_indices]
            return save_global_style(indices, room_type, style_theme)
            
        def process_save_local(image_indices, room_type, style_theme, mask_label):
            # 从选择的方案中提取索引号
            indices = [int(idx.split(" ")[1]) for idx in image_indices]
            return save_local_style(indices, room_type, style_theme, mask_label)
        
        # 全局风格调整保存按钮事件
        save_btn.click(
            process_save_global,
            inputs=[save_image_index, room_type, style_theme],
            outputs=[save_status]
        )
        
        # 局部风格调整保存按钮事件
        save_btn_local.click(
            process_save_local,
            inputs=[save_image_index_local, room_type_local, style_theme_local, mask_label_local],
            outputs=[save_status_local]
        )
    
    return app

# 启动应用
if __name__ == "__main__":
    app = create_interface()
    app.launch()
