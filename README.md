# CAV 自包含认证（B 方案）

本项目提供一种**自包含、脆弱型**的模型完整性认证方案。认证信息直接嵌入模型参数中，验证时只需要 **checkpoint + 密钥**，不依赖任何外部 meta 文件。

核心特性：
- 自包含：无需额外 json/meta 或签名文件
- 强完整性：任意权重 bit 变化都会 FAIL
- 可复现：密钥驱动的探针与嵌入映射
- 低载荷：固定 64 字节 payload 写入 LSB

## 快速开始

1) 安装依赖
```bash
pip install -r requirements.txt
```

2) 训练模型
```bash
python train_classifier.py --dataset cifar10 --model resnet20 --data_root /path/to/datasets --epochs 20 --out_dir runs
```

3) 生成密钥（32 字节 hex）
```bash
python - <<'PY'
import os, binascii
print(binascii.hexlify(os.urandom(32)).decode())
PY
```

4) 嵌入自包含认证
```bash
python embed_auth_b.py \
  --ckpt runs/cifar10_resnet20/best.pt \
  --out runs/cifar10_resnet20/with_auth.pt \
  --key_hex <YOUR_KEY_HEX> \
  --device cpu
```

5) 验证（PASS/FAIL）
```bash
python verify_auth_b.py \
  --ckpt runs/cifar10_resnet20/with_auth.pt \
  --key_hex <YOUR_KEY_HEX> \
  --device cpu
```

## 原理说明（自包含 B 方案）

1) 参数域承诺（h_param）
- 按 module_id（名字最后一个点之前的前缀）分组 state_dict。
- 对每个 module_id 的所有条目，按 name/dtype/shape/bytes 稳定序列化并与密钥一起哈希。
- 任意参数 bit 变化都会导致 h_param 变化。

2) CAV 行为承诺（h_cav）
- 由密钥生成确定性合成探针（不依赖数据集）。
- 使用密钥导出的固定目标类进行反向传播。
- 按梯度幅值选 top-m 通道。
- 取 SVD 特征与激活均值/方差并量化，再与密钥一起哈希。

3) Layer Leaf 与 Merkle Root
- leaf = SHA256(key | "LEAF" | module_id | h_param | h_cav)
- 按 module_id 排序后构造 Merkle root。

4) Payload 与 LSB 嵌入
- 固定 64 字节 payload：magic + 版本 + cfg 字段 + root + CRC32。
- 使用**多载体随机映射**将 payload bits 写入多个 float32 参数的 LSB。
- 不使用 ECC / 冗余投票；验证严格比较 root。

5) 验证
- 从 LSB 提取 payload，校验 magic 与 CRC32。
- 从 payload 还原 cfg 并重算 root。
- 仅当 root_calc == root_emb 时 PASS。

## 常用参数

`embed_auth_b.py` 常用参数：
- `--probe_n`、`--probe_h`、`--probe_w`
- `--top_m`、`--sv_k`、`--quant_scale`
- `--device cpu`（建议固定 CPU 以确保确定性）

## 工程结构

```
src/
  auth/
    cav_selfcontained_auth.py   # 自包含方案（B）
  data/
  engine/
  models/
  utils/
embed_auth_b.py                 # 嵌入自包含认证
verify_auth_b.py                # 自包含验证
train_classifier.py
```

## 注意事项

- 密钥必须保密，否则攻击者可重算并重新嵌入 root。
- 方案为脆弱水印设计，任何权重改动都会 FAIL。
- 不使用 ECC 或投票机制，以避免小篡改被“纠回”。
