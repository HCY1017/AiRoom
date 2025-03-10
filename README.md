# DesignGenie - AI驱动的室内设计生成工具

DesignGenie是一个基于人工智能的室内设计生成工具，利用ControlNet和Stable Diffusion模型，可以根据输入的房间图像生成多种风格的室内设计方案。

## 项目概述

本项目提供了两种主要功能：
1. **整体设计生成**（whole.py）：使用ControlNet控制整体布局，Stable Diffusion调整全局风格
2. **局部设计修改**（inpaint.py）：使用ControlNet控制整体布局，Stable Diffusion进行局部区域的修改（Inpainting）

## 功能特点

- 自动识别房间中的不同区域（使用语义分割技术）
- 保持原始房间的结构和布局（通过ControlNet的线条检测）
- 根据文本提示词生成符合描述的设计风格
- 支持生成多个设计变体，便于比较和选择
- 可以选择性地修改房间中的特定区域（如家具、墙壁等）

## 安装与环境配置

本项目依赖于多个深度学习库和工具。我们提供了完整的环境配置文件`environment.yml`，可以使用Conda快速创建所需环境：

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate designgenie
```

主要依赖包括：
- PyTorch
- Diffusers
- Transformers
- ControlNet-aux
- Matplotlib

## 使用方法

### 1. 整体设计生成

运行`whole.py`脚本来生成整个房间的设计方案：

```bash
python whole.py
```

默认情况下，脚本会：
1. 加载示例输入图像（sample_input.png）
2. 使用Mask2Former进行语义分割
3. 使用MLSD检测器生成线条控制图像
4. 根据预设的提示词生成4个不同的设计方案

### 2. 局部设计修改

运行`inpaint.py`脚本来修改房间中的特定区域：

```bash
python inpaint.py
```

此脚本会：
1. 加载示例输入图像
2. 进行语义分割并生成区域掩码
3. 仅修改选定的区域（由`mask_ID`参数控制）
4. 生成4个不同的设计变体

## 自定义设计

您可以通过修改脚本中的以下参数来自定义生成结果：

1. **输入图像**：修改`img_src`变量指向您自己的图像
2. **掩码ID**：修改`mask_ID`变量选择不同的区域进行修改
3. **提示词**：修改`prompt`变量来描述您想要的设计风格
4. **负面提示词**：修改`negative_prompt`变量来避免不想要的元素

## 示例输出

运行脚本后，将显示一个包含以下内容的图像网格：
- 原始输入图像
- 控制图像（结构引导）
- 选定区域的掩码
- 4个生成的设计变体

## 技术细节

本项目结合了多种先进的AI技术：

1. **语义分割**：使用Mask2Former模型识别房间中的不同区域
2. **线条检测**：使用MLSD检测器提取房间的结构线条
3. **ControlNet**：确保生成的设计保持原始房间的结构
4. **Stable Diffusion**：根据文本提示词生成符合描述的设计风格
5. **图像修复（Inpainting）**：仅修改选定区域，保持其他部分不变

## 注意事项

- 本项目需要较大的GPU内存（建议至少8GB）
- 首次运行时会下载预训练模型，可能需要一些时间
- 生成过程可能需要几分钟，取决于您的硬件配置

## 许可证

[请在此处添加您的许可证信息]

## 致谢

本项目基于以下开源项目：
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [Diffusers](https://github.com/huggingface/diffusers)
