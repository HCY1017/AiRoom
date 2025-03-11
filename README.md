# AiRoom - AI辅助室内设计工具

## 项目简介

AiRoom是一个基于AI技术的室内设计辅助工具，通过结合ControlNet和Stable Diffusion模型，实现对室内场景的全局风格调整和局部区域风格定制。该工具提供了直观的交互式界面，使用户能够轻松地对室内设计进行创意探索和风格转换。

## 功能特点

- **全局风格调整**：使用ControlNet保持原始空间布局的同时，通过Stable Diffusion调整整体风格
- **局部风格调整**：针对特定区域（如墙壁、地板、家具等）进行风格定制，保持其他区域不变
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
   - 首先加载模型
   - 选择功能模式（全局风格调整或局部风格调整）
   - 上传室内场景图片或使用示例图片
   - 根据需要调整参数
   - 生成并查看设计方案

## 功能详解

### 全局风格调整

全局风格调整功能允许用户保持原始空间布局的同时，改变整个场景的设计风格。用户可以：

- 输入详细的风格描述提示词
- 选择房间类型（卧室、客厅、厨房等）
- 选择风格主题（现代、北欧、工业风等）
- 调整推理步数和引导比例等参数

### 局部风格调整

局部风格调整功能允许用户针对场景中的特定区域进行风格定制，而保持其他区域不变。用户可以：

- 从下拉菜单中选择要调整的区域
- 查看所选区域的掩码预览（红色半透明覆盖）
- 输入针对该区域的风格描述提示词
- 生成保持整体结构的局部风格变化

## 项目结构

- `app.py`：主应用程序，包含Gradio界面和核心功能
- `gloab.py`：全局风格调整功能的独立实现
- `inpaint.py`：局部风格调整功能的独立实现
- `download_resources.py`：下载必要模型和资源的工具脚本
- `requirements.txt`：项目依赖列表
- `resources/`：存放模型、图像和标签数据的目录

## 技术实现

项目使用了多种先进的AI模型和技术：

- **Mask2Former**：用于场景语义分割，识别不同功能区域
- **ControlNet (MLSD)**：保持原始场景的结构和布局
- **Stable Diffusion**：生成符合提示词描述的图像内容
- **Gradio**：构建直观的用户界面

## 注意事项

- 首次运行时需要下载较大的模型文件（约10GB），请确保有足够的磁盘空间和稳定的网络连接
- 生成过程可能需要较长时间，取决于您的硬件配置
- 为获得最佳效果，建议使用清晰的室内场景照片作为输入

## 许可证

[在此添加您的许可证信息]

## 致谢

本项目基于以下开源项目和模型：

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [Gradio](https://github.com/gradio-app/gradio)

## 联系方式

[在此添加您的联系信息]
