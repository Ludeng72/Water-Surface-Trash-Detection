# 水面垃圾检测系统 (Water Surface Trash Detection)

## 项目简介
基于 YOLOv8 的水面漂浮物检测项目，针对河流、湖泊等场景的 9 类常见垃圾进行识别。

## 技术栈
- 算法：YOLOv8 (SEDS 优化版) + PyTorch
- 后端：FastAPI + Uvicorn
- 前端：HTML + JavaScript

## 数据集说明
使用了包含 5000 张图像的公开数据集，涵盖 9 个类别（易拉罐、树叶、树枝、草、瓶子、塑料盒、牛奶盒、塑料袋、纸）。
**数据划分**：本版本重点在于跑通全流程。目前使用的是训练集和验证集（未独立划分测试集），但在后续迭代中计划将按 8:1:1 补充测试集，以确保模型的泛化评估。

## 训练结果
- 训练环境：RTX 5060 Laptop GPU (CUDA 12.8) + PyTorch Nightly
- 评估指标 (20 Epochs)：
  - mAP50: 0.494
  - 易拉罐 (can) 表现最佳，mAP50 达到 0.745
  - 环保植被类（草、树叶）因背景混淆导致指标较低，后续计划引入注意力机制优化

## 部署体验
1. 安装依赖：`pip install -r requirements.txt`
2. 启动后端：`uvicorn app:app --reload`
3. 打开 `index.html` 即可通过网页上传图片进行实时检测。

## 📸 效果展示
<img width="321" height="321" alt="download" src="https://github.com/user-attachments/assets/710dc8ce-dd1c-4c07-9729-490ba6a628e0" />
