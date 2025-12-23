# CAV-MerkleAuth (工程化可复现白盒完整性认证)

本工程实现 **CAV-MerkleAuth**：一种面向 CNN 分类模型的白盒完整性认证（integrity authentication）方法，支持：
- **结构可验证**：面对 *dummy channel / 结构混淆* 仍可计算指纹并完成验证；
- **全局强绑定**：层指纹哈希 + Merkle Root；
- **自包含认证**：将 Root 位流嵌入到模型参数的 float32 LSB；
- **定位/恢复（基础版）**：块级耦合校验 + Root 冗余多数投票恢复。

工程同时提供训练/评测脚本，默认支持数据集：**CIFAR-10、CIFAR-100、GTSRB、MNIST**，
模型：**LeNet、ResNet18、ResNet20、VGG16**。

---

## 1. 环境依赖
```bash
pip install -r requirements.txt
```

## 2. 数据集目录约定
假设 `--data_root /path/to/datasets`，则应包含（与你截图一致）：
```
/path/to/datasets/
  cifar-10-batches-py/
  cifar-100-python/
  MNIST/            (torchvision 结构：MNIST/raw, MNIST/processed)
  gtsrb/            (支持常见两种结构，见下文)
```

### GTSRB 支持的两种结构
1) 官方结构：`gtsrb/Final_Training/Images/...` + CSV  
2) ImageFolder 结构：`gtsrb/train/<class_id>/*.png`，`gtsrb/test/<class_id>/*.png`

---

## 3. 快速开始（端到端）
### 3.1 训练一个分类模型
```bash
python train_classifier.py --dataset cifar10 --model resnet18 --data_root /path/to/datasets --epochs 20 --out_dir runs
```

### 3.2 对训练好的模型嵌入认证信息（CAV-MerkleAuth）
```bash
python embed_auth.py --ckpt runs/cifar10_resnet18/best.pt --out runs/cifar10_resnet18/with_auth.pt
```

### 3.3 验证完整性（含层级定位/块级定位/恢复）
```bash
python verify_auth.py --ckpt runs/cifar10_resnet18/with_auth.pt --meta runs/cifar10_resnet18/auth_meta.json --data_root /path/to/datasets
```

### 3.4 进行篡改攻击并验证失败（示例：随机噪声篡改）
```bash
python attack_and_verify.py --ckpt runs/cifar10_resnet18/with_auth.pt --meta runs/cifar10_resnet18/auth_meta.json --attack random_noise --rate 0.01
```

---

## 4. 工程结构
```
cav_merkle_auth_project/
  src/
    auth/         # CAV-MerkleAuth 核心
    data/         # 数据集加载
    models/       # 模型定义
    engine/       # 训练/评测循环
    utils/        # 日志、seed、ckpt、通用函数
  train_classifier.py
  embed_auth.py
  verify_auth.py
  attack_and_verify.py
```

---

## 5. 说明
- 这是“基础可复现版”。论文增强版可在 `src/auth/cav_merkle_auth.py` 上进一步加入：
  - 激活梯度×激活（IG）重要性，强化 dummy channel 过滤；
  - 载体选择：优先选择低敏感权重作为 LSB carrier，进一步降低精度损失；
  - 更强纠错（LDPC/RS）与更稳的量化策略。
