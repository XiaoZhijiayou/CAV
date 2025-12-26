###### 其中json部分的解释

```json 
python verify_auth_b.py \
  --ckpt "/root/autodl-tmp/CAV-MerkleAuth/checkpoints/best_resnet18_cifar10_r18_c10_auth_v2_pruned_l30conv1.pth" \
  --key_hex "efb0d0213a3eb4eea60befd48f5c647184dec84917047de999de554547b895a6" \
  --device cpu
FAIL
{
  "ok": false,
  "magic_ok": true,
  "crc_ok": true,
  "version": 1,
  "root_emb_hex": "3067019bb3e40c08db52b2d46a48a77f093f1e87dcb621951303e6d0a01edf51",
  "root_calc_hex": "ecd28178e203a53a91cca6361e3e6286940f9a9c29fbeb3b14671268f98d06b9",
  "carrier_params": [
    "linear.weight",
    "bn1.bias",
    "bn1.weight",
    "conv1.weight",
    "layer1.0.bn1.bias",
    "layer1.0.bn1.weight",
    "layer1.0.bn2.bias",
    "layer1.0.bn2.weight",
    "layer1.0.conv1.weight",
    "layer1.0.conv2.weight",
    "layer1.1.bn1.bias",
    "layer1.1.bn1.weight",
    "layer1.1.bn2.bias",
    "layer1.1.bn2.weight",
    "layer1.1.conv1.weight",
    "layer1.1.conv2.weight",
    "layer2.0.bn1.bias",
    "layer2.0.bn1.weight",
    "layer2.0.bn2.bias",
    "layer2.0.bn2.weight",
    "layer2.0.conv1.weight",
    "layer2.0.conv2.weight",
    "layer2.0.shortcut.0.weight",
    "layer2.0.shortcut.1.bias",
    "layer2.0.shortcut.1.weight",
    "layer2.1.bn1.bias",
    "layer2.1.bn1.weight",
    "layer2.1.bn2.bias",
    "layer2.1.bn2.weight",
    "layer2.1.conv1.weight",
    "layer2.1.conv2.weight",
    "layer3.0.bn1.bias",
    "layer3.0.bn1.weight",
    "layer3.0.bn2.bias",
    "layer3.0.bn2.weight",
    "layer3.0.conv1.weight",
    "layer3.0.conv2.weight",
    "layer3.0.shortcut.0.weight",
    "layer3.0.shortcut.1.bias",
    "layer3.0.shortcut.1.weight",
    "layer3.1.bn1.bias",
    "layer3.1.bn1.weight",
    "layer3.1.bn2.bias",
    "layer3.1.bn2.weight",
    "layer3.1.conv1.weight",
    "layer3.1.conv2.weight",
    "layer4.0.bn1.bias",
    "layer4.0.bn1.weight",
    "layer4.0.bn2.bias",
    "layer4.0.bn2.weight",
    "layer4.0.conv1.weight",
    "layer4.0.conv2.weight",
    "layer4.0.shortcut.0.weight",
    "layer4.0.shortcut.1.bias",
    "layer4.0.shortcut.1.weight",
    "layer4.1.bn1.bias",
    "layer4.1.bn1.weight",
    "layer4.1.bn2.bias",
    "layer4.1.bn2.weight",
    "layer4.1.conv1.weight",
    "layer4.1.conv2.weight",
    "linear.bias"
  ],
  "capacity_bits": 93258,
  "loc": {
    "ok": false,
    "mismatched_modules": [
      "layer1.1.conv1",
      "layer2.0.shortcut.0",
      "layer3.0.conv1",
      "layer3.0.conv2",
      "layer3.1.conv1",
      "layer3.1.conv2",
      "layer4.0.conv1",
      "layer4.0.conv2",
      "layer4.0.shortcut.0",
      "layer4.1.conv1",
      "layer4.1.conv2",
      "linear"
    ],
    "missing_modules": [],
    "extra_modules": []
  },
  "loc_param": {
    "ok": false,
    "mismatched_modules": [
      "layer3.0.conv1"
    ],
    "missing_modules": [],
    "extra_modules": []
  }
}
```

- ok: 总体验证是否通过（魔数+CRC+root 是否一致）。剪枝后 root 改了，所以是 false。

- magic_ok: 从权重 LSB 提取出来的 payload 魔数是否匹配（格式正确）。

- crc_ok: payload CRC 校验是否通过（payload 没坏）。

- version: payload 版本号（来自嵌入的 payload，不是 cav_loc 版本）。

- root_emb_hex: 嵌入时写进去的 Merkle root（原模型指纹）。

- root_calc_hex: 当前模型重新计算的 Merkle root（被攻击后）。

- carrier_params: 用来承载 LSB 的参数列表（载体集合）。

- capacity_bits: 载体总容量（可用 LSB 位数）。

- loc: 叶子级定位（参数承诺 + CAV 指纹）

    - mismatched_modules: 叶子哈希不一致的模块列表（包含 CAV 指纹，敏感度高，会“扩散”到多层）。
    - missing_modules/extra_modules: 模块结构变化时才会出现。
- loc_param: 参数级定位（只看参数承诺哈希）
    - mismatched_modules: 只受权重变化影响，单层剪枝时应只出现目标层。

