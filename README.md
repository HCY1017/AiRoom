# AiRoom - AI辅助室内设计工具

## 项目简介

AiRoom是一个基于AI技术的室内设计辅助工具，通过结合ControlNet和Stable Diffusion模型，实现对室内场景的全局风格调整和局部区域风格定制。该工具提供了直观的交互式界面，使用户能够轻松地对室内设计进行创意探索和风格转换，并支持相似图像搜索功能，帮助用户发现灵感。

## 功能特点

- **全局风格调整**：使用ControlNet保持原始空间布局的同时，通过Stable Diffusion调整整体风格
- **局部风格调整**：针对特定区域（如墙壁、地板、家具等）进行风格定制，保持其他区域不变
- **相似图像搜索**：基于CLIP和FAISS实现的高效图像相似性搜索，帮助用户发现相似设计方案
- **交互式界面**：基于Gradio构建的用户友好界面，支持实时预览和参数调整
- **多方案生成**：每次生成多个设计方案供用户选择，以2x2网格形式展示
- **区域智能识别**：自动分析图像中的不同功能区域，无需手动标注

## 安装说明

### 环境要求

- Python 3.8+
- CUDA支持的GPU (推荐8GB+显存)

### 安装步骤

1. 克隆本仓库到本地：

```bash
git clone https://github.com/yourusername/AiRoom.git
cd AiRoom
```

2. 创建并激活虚拟环境（推荐）：

```bash
# 使用Conda创建虚拟环境
conda create -n Airoom python=3.10
conda activate Airoom

# 或使用venv创建虚拟环境
python -m venv Airoom
# Windows激活
Airoom\Scripts\activate
# Linux/Mac激活
source Airoom/bin/activate
```

3. 安装依赖包：

```bash
# 安装PyTorch（根据您的CUDA版本选择适当的命令）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

4. 下载必要资源：

```bash
python download_resources.py
```

## 使用指南

1. 启动应用：

```bash
python app.py
```

2. 在浏览器中访问显示的本地URL（通常为 http://127.0.0.1:7860）

3. 使用流程：
   - 首先点击"加载模型"按钮，等待所有模型加载完成
   - 选择功能模式（通过顶部选项卡：全局风格调整、局部风格调整或相似图像搜索）
   - 上传室内场景图片或使用示例图片
   - 点击"分析图像结构"按钮处理输入图像
   - 根据需要调整参数
   - 点击"生成设计方案"按钮创建新设计
   - 从生成的多个设计方案中选择喜欢的结果
   - 可选择保存设计方案供后续参考或搜索

## 功能详解

### 全局风格调整

全局风格调整功能允许用户保持原始空间布局的同时，改变整个场景的设计风格。用户可以：

- 输入详细的风格描述提示词（可从预设列表中选择或自定义）
- 选择房间类型（卧室、客厅、厨房等）
- 选择风格主题（现代、北欧、工业风等）
- 调整推理步数（影响生成质量和时间）
- 调整引导比例（影响生成结果对提示词的遵循程度）
- 同时生成4个不同的设计方案进行比较
- 选择并保存喜欢的设计方案

工作原理：
- 使用MLSD检测器提取房间的线条结构，生成控制图像
- ControlNet确保生成的图像保持原始空间布局和结构
- Stable Diffusion根据提示词和控制图像生成符合要求的设计风格

### 局部风格调整

局部风格调整功能允许用户针对场景中的特定区域进行风格定制，而保持其他区域不变。用户可以：

- 从下拉菜单中选择要调整的区域（墙壁、地板、家具等）
- 查看所选区域的掩码预览（红色半透明覆盖显示选中区域）
- 输入针对该区域的风格描述提示词
- 调整区域变化的强度和细节
- 生成保持整体结构的局部风格变化

工作原理：
- 使用Mask2Former模型进行语义分割，识别图像中的不同功能区域
- 将识别的区域转换为掩码，供用户选择
- 结合ControlNet和Stable Diffusion Inpainting进行局部区域的风格调整
- 保持未选中区域不变，只修改选中区域的风格

### 相似图像搜索

相似图像搜索功能利用CLIP模型和FAISS索引，帮助用户查找与参考图像风格相似的设计方案。用户可以：

- 上传参考图像
- 设置搜索结果数量（2-8个）
- 查看以2x2网格布局展示的相似图像结果
- 查看每个结果的相似度百分比
- 通过"重建图像索引"按钮更新索引，包含新生成的设计方案

工作原理：
- 使用CLIP模型提取图像的语义特征向量
- FAISS索引存储所有已生成设计方案的特征向量
- 搜索时计算查询图像与索引中所有图像的余弦相似度
- 返回相似度最高的图像作为结果

## 项目结构

- `app.py`：主应用程序，包含Gradio界面和核心功能实现（全局风格调整、局部风格调整、相似图像搜索）
- `download_resources.py`：下载必要模型和资源的工具脚本
- `requirements.txt`：项目依赖列表
- `resources/`：存放模型、图像和标签数据的目录
  - `models/`：存储AI模型（Mask2Former、ControlNet、Stable Diffusion等）
  - `images/`：存储示例和生成的图像
  - `labels/`：存储标签数据（如ADE20K数据集标签）
  - `output/`：存储生成的设计方案
    - `global_style/`：全局风格调整生成的图像
    - `local_style/`：局部风格调整生成的图像
  - `features/`：存储图像特征和索引文件（用于相似图像搜索）
    - `image_features.index`：FAISS索引文件
    - `image_metadata.pkl`：图像元数据文件

## 技术实现

项目使用了多种先进的AI模型和技术：

- **Mask2Former**：用于场景语义分割，识别不同功能区域（如墙壁、地板、家具等）
- **ControlNet (MLSD)**：保持原始场景的结构和布局，通过线条检测提供控制指导
- **Stable Diffusion**：生成符合提示词描述的图像内容
- **Stable Diffusion Inpainting**：针对特定区域进行图像修复和风格转换
- **CLIP**：提取图像特征，用于相似性搜索和语义理解
- **FAISS**：高效的向量相似性搜索库，支持大规模图像检索
- **Gradio**：构建直观的用户界面，支持交互式操作和实时预览
- **PyTorch**：深度学习框架，支持GPU加速的模型推理

模型加载策略：
- 使用`torch.float16`精度减少内存占用
- 实现模型CPU卸载以优化内存使用
- 支持xformers内存优化（如果安装）
- 从本地缓存加载模型，避免重复下载

## 注意事项

- 首次运行时需要下载较大的模型文件（约10GB），请确保有足够的磁盘空间和稳定的网络连接
- 生成过程可能需要较长时间，取决于您的硬件配置（推荐使用NVIDIA GPU）
- 为获得最佳效果，建议使用清晰的室内场景照片作为输入
- 相似图像搜索功能需要先生成并保存一些设计方案才能有效工作
- 调整推理步数可以平衡生成质量和速度，通常20-30步可以获得不错的结果
- 调整引导比例可以控制生成结果的创意程度，较高的值（7-9）会更严格遵循提示词

## 许可证

[在此添加您的许可证信息]

## 致谢

本项目基于以下开源项目和模型：

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [CLIP](https://github.com/openai/CLIP)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gradio](https://github.com/gradio-app/gradio)

## 联系方式

[在此添加您的联系信息]
