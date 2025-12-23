# CAV-MerkleAuth (工程化可复现白盒完整性认证)

本工程实现 **CAV-MerkleAuth**：一种面向 CNN 分类模型的白盒完整性认证（integrity authentication）方法，支持：
- **结构可验证**：面对 *dummy channel / 结构混淆* 仍可计算指纹并完成验证；
- **全局强绑定**：层指纹哈希 + Merkle Root；
- **自包含认证**：将 Root 位流嵌入到模型参数的 float32 LSB；
- **签名防伪**：对 `auth_meta` 进行 Ed25519 签名，防止白盒伪造；
- **纠错鲁棒**：ECC 编码（默认 Hamming 15,11）+ Root 冗余多数投票；
- **分散嵌入**：多载体参数 + 密钥驱动随机映射；
- **配置绑定**：`cfg_hash` 绑定关键超参与层列表。

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

### 3.2 准备认证密钥与签名密钥
主密钥用于指纹与嵌入（32 bytes hex，示例）：
```bash
python - <<'PY'
import os, binascii
print(binascii.hexlify(os.urandom(32)).decode())
PY
```
生成 Ed25519 密钥对（示例）：
```bash
openssl genpkey -algorithm ed25519 -out ed25519_priv.pem
openssl pkey -in ed25519_priv.pem -pubout -out ed25519_pub.pem
```

### 3.3 对训练好的模型嵌入认证信息（CAV-MerkleAuth）
```bash
python embed_auth.py \
  --ckpt runs/cifar10_resnet18/best.pt \
  --out runs/cifar10_resnet18/with_auth.pt \
  --key_hex <YOUR_KEY_HEX> \
  --privkey ed25519_priv.pem
```

### 3.4 验证完整性（含层级定位与 ECC 纠错）
```bash
python verify_auth.py \
  --ckpt runs/cifar10_resnet18/with_auth.pt \
  --meta runs/cifar10_resnet18/auth_meta.json \
  --key_hex <YOUR_KEY_HEX> \
  --pubkey ed25519_pub.pem
```

### 3.5 进行篡改攻击并验证失败（示例：随机噪声篡改）
```bash
python attack_and_verify.py \
  --ckpt runs/cifar10_resnet18/with_auth.pt \
  --meta runs/cifar10_resnet18/auth_meta.json \
  --key_hex <YOUR_KEY_HEX> \
  --pubkey ed25519_pub.pem \
  --attack random_noise --rate 0.01
```

> 若仅做调试，可在嵌入时使用 `--no_sign`，验证时加 `--allow_unsigned`。

---

## 4. 认证流程（v2）
1) **指纹生成**：probe 前向 + 目标类反传，选 top‑m 通道；
2) **特征拼接**：权重谱特征（SVD）+ 激活均值/方差 + 密钥驱动权重采样；
3) **层哈希 + Merkle Root**：层指纹哈希后构建 Merkle Root；
4) **Payload 构造**：`header + cfg_hash + root + redundancy`；
5) **ECC 编码**：默认 Hamming(15,11) 提升鲁棒性；
6) **分散嵌入**：多个参数张量，密钥驱动随机映射写入 LSB；
7) **签名**：对 `auth_meta` 进行 Ed25519 签名；
8) **验证**：验签 → 提取 bits → ECC 解码 → header/cfg_hash 校验 → root 对比 → 层级定位。

---

## 5. 工程结构
```
cav_merkle_auth_project/
  src/
    auth/         # CAV-MerkleAuth 核心
    data/         # 数据集加载
    models/       # 模型定义
    engine/       # 训练/评测循环
    utils/        # 日志、seed、ckpt、通用函数（含签名/密钥工具）
  train_classifier.py
  embed_auth.py
  verify_auth.py
  attack_and_verify.py
```

---

## 6. 说明
- 本版本引入外部密钥与签名，白盒攻击者无法在未知私钥的情况下伪造 `auth_meta`。
- 若模型被重新量化/裁剪/转 FP16，嵌入信息可能损坏，需提高 ECC 或改用更稳健载体。
- 论文增强版可在 `src/auth/cav_merkle_auth.py` 上进一步加入：
  - 激活梯度×激活（IG）重要性，强化 dummy channel 过滤；
  - 载体选择：优先选择低敏感权重作为 LSB carrier，进一步降低精度损失；
  - 更强纠错（LDPC/RS）与更稳的量化策略。
