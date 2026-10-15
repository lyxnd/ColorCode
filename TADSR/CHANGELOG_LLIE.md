# TADSR → LLIE 修改记录

原则：**不改变 t 的语义**（t 仍然控制保真度与真实感之间的权衡）。Time-Aware VAE、UNet LoRA、Time-Aware VSD 和损失函数都没有改动。
所有改动处都标了 `[LLIE]` 注释，可以全局搜索。下面的行号都是**修改后文件**的行号。

---

## 2026-09-30 第 1 轮：修掉 SR 专用设置 / bug，使其能直接用于 LLIE

### 1. `dataloaders/paired_dataset.py`
| 行号 | 改动 |
|---|---|
| 28–45 | 列表解析：每行 `GT  LQ  tag`。优先按 **Tab** 切分（路径里可以有空格），否则按空白切分且最多切 2 次，保留 tag 里的空格。跳过空行；没有 tag 时 prompt 为空串。文件以 utf-8 打开。 |
| 58–60 | 新增 `self.crop_size = args.resolution`，要求是 8 的倍数。 |
| 62–76 | 新增 `_paired_random_crop`：GT 和 LQ 用**同一位置**随机裁剪；短边小于裁剪尺寸时先等比放大。原代码完全没有裁剪。 |
| 78–84 | 新增 `_paired_augment`：GT 和 LQ 同步做水平/垂直翻转。 |
| 86–103 | `__getitem__`：**删除了原来 SR 的 bicubic ×4 退化分支**，所以 `--deg_prob` 不再起作用，始终读取配对的低光图。LQ 只在和 GT 尺寸不同时才 resize。裁剪和翻转只在 `split=="train"` 时执行。 |

### 2. `get_tag.py`
| 行号 | 改动 |
|---|---|
| 4–5 | 在文件头说明里写明新的输出格式。 |
| 30 | 新增 `--tag_source {hr,lr}`，默认 `hr`，即**在 GT（正常光）图上打 tag**。暗图上 RAM 基本识别不出内容。 |
| 35–39 | 支持 png/jpg/jpeg/bmp 等后缀（原来只读 png），并打印找到的图片数量。 |
| 41–43 | 先定义 device 再加载模型，模型放到指定的 `--gpu_id` 上（原来固定是 `cuda`）。 |
| 57–65 | 找不到 GT 时打印警告；打 tag 用的图由 `--tag_source` 决定；读图时加了 `.convert('RGB')`；tag 里的 Tab 和换行被清掉。 |
| 69–70 | **修正列顺序 bug**：原来写的是 `LQ HR tag`，而数据集按 `GT LQ tag` 读取，会把暗图当成 GT。现在输出 `GT<TAB>LQ<TAB>tag`。 |
| 46, 71, 76 | 统计并打印实际写入的配对数量。 |

### 3. `test_tadsr.py`
| 行号 | 改动 |
|---|---|
| 28–37 | 新增 `brighten_for_tagging`：用 gamma 把输入图的平均亮度拉到 `--ram_brighten`，**提亮后的图只用于 RAM 打 tag**。 |
| 43 | `get_validation_prompt` 改为对提亮后的副本打 tag。返回的 `lq`（送进网络的图）仍是原始暗图。 |
| 61–62 | `--process_size` 默认值 512 → **0**，按原分辨率处理，不再把 LOL 600×400 先放大到 768×512。 |
| 63 | `--upscale` 默认值 4 → **1**。 |
| 64–65 | `--align_method` 默认值 `adain` → **`nofix`**。adain 和 wavelet 会把输出的亮度和颜色对齐回暗输入，LLIE 不能用。 |
| 66 | 新增 `--ram_brighten`，默认 0.4；设为 ≤0 时关闭提亮。 |
| 90–92 | 输入目录支持 png/jpg/jpeg/bmp 等后缀。 |
| 162–166 | **修正 bug**：原来的 `output_pil.resize(...)` 没有赋值，等于没做。现在总是把输出还原到 `原图尺寸 × upscale`，保证和 GT 逐像素对齐，也抵消了为凑 8 的倍数做的 resize。 |
| 169–170 | 输出统一存成 `.png`。 |

### 4. `tadsr.py`
| 行号 | 改动 |
|---|---|
| 482–483 | **修正 bug**：`TADSR_test.forward` 分块分支里用到了未定义的 `self.timesteps`，latent 边长超过 96（图片边长超过约 768px）时会报错。改为使用传入的 `timesteps`。 |

### 5. `scripts/train_tadsr.sh`
| 行号 | 改动 |
|---|---|
| 6 | 新增 `--resolution=384`。LOL 图高只有 400，裁剪尺寸必须是 8 的倍数。 |

### 6. `scripts/test_tadsr.sh`
| 行号 | 改动 |
|---|---|
| 7–12 | 显式写出 `--upscale 1 --align_method nofix --process_size 0 --ram_brighten 0.4 --timesteps 500`。 |

### 没有改动
`train_tadsr.py`、Time-Aware VAE（`diffusers/models/autoencoders/*`）、UNet，以及全部损失函数（随 t 变化的模糊 L2、LPIPS、Time-Aware VSD、diff loss）。
`train_tadsr.py` 里的 `--deg_prob` 参数还在，但数据集已经不再读取它。

### 使用流程
1. 生成列表：`python get_tag.py --ram_path preset/weights/ram_swin_large_14m.pth --hr_path <GT目录> --lr_path <低光目录> --out_txt preset/dataset_list.txt`。GT 和低光图的**文件名需要一一对应**。
2. 训练：`bash scripts/train_tadsr.sh`。如果想从 TADSR 的 SR 权重开始微调，加上 `--pretrained_lora_path preset/weights/tadsr.pkl`。
3. 测试：`bash scripts/test_tadsr.sh`。可以扫描 `--timesteps`（如 100/300/500/700/900）：输出的平均亮度应该基本不变，细节随 t 增加。

### 验证情况
- 所有改动过的 .py 都通过了 `py_compile`。
- 数据集用 600×400 的合成配对图实测过：输出 3×384×384；Tab 格式、带空格的路径、CRLF 行尾都能正确解析；30 次随机裁剪加翻转后，GT 和 LQ 逐像素对齐（误差 0，错配的对照组误差为 25 个灰度级）。
- `brighten_for_tagging` 实测：平均亮度 0.02/0.08/0.2 的输入被提到约 0.38；亮度 ≥0.4 的输入保持不变；关闭时原样返回。
- **没有在 GPU 上跑完整的训练或推理**（这个环境里没有模型权重），需要你本地跑一次确认。

---

## 2026-10-08 第 2 轮：到 max_train_steps 自动停止 + 验证并保存 best

> 这一轮在 `dataloaders/paired_dataset.py` 里插入了 4 行，第 1 轮记录中该文件的行号要顺延：第 22 行之后 +2，第 52 行之后 +4。

### 1. `train_tadsr.py`
| 行号 | 改动 |
|---|---|
| 127–137 | 新增参数：`--save_best`（**默认开启**，用 `--no-save_best` 关闭）、`--val_txt_path`（验证集列表，格式和训练列表相同）、`--validation_steps`（默认 500）、`--val_timesteps`（默认 `500`，可写多个如 `200,500,800`，指标取平均）、`--best_metric`（`psnr` 或 `lpips`，默认 `psnr`）、`--val_num_images`（只用前 N 张验证图，默认 0 = 全部）。 |
| 168–220 | 新增 `run_validation`：在验证集上**整图**推理（裁到 8 的倍数），计算平均 PSNR 和 LPIPS。输出先量化到 8bit 再算，和存成 png 后算的结果一致。验证时固定随机种子，结束后恢复训练的随机状态和 train 模式，不影响训练过程。 |
| 330–339 | 创建验证集（`split="val"`，不裁剪不翻转）。开启了 `--save_best` 但没有可用的验证列表时，打印 WARNING 并跳过 best 保存，训练照常进行。 |
| 367–393 | 新增 best 记录变量和 `validate_and_save_best`：验证 → 写入 TensorBoard（`val/psnr`、`val/lpips`、`val/best_*`）→ 指标更好时覆盖 `checkpoints/model_best.pkl`，并把步数和指标写到 `checkpoints/best.txt`。 |
| 466 | 记录最近一次保存定期 checkpoint 的步数，避免结束时重复保存。 |
| 470–472 | 每 `--validation_steps` 步在主进程上验证一次。 |
| 474–479 | **到达 `--max_train_steps` 后跳出两层循环**。原代码只把这个参数用在进度条和学习率调度上，训练不会停。 |
| 481–491 | 训练结束时：保存最终权重 `model_{step}.pkl`，做最后一次验证，打印 best 结果，关闭进度条和日志。 |

### 2. `dataloaders/paired_dataset.py`
| 行号 | 改动 |
|---|---|
| 17, 22–24 | `__init__` 新增可选参数 `txt_path`，验证集用它传入自己的列表；不传时和原来一样读 `--txt_path` 的第一个文件。 |
| 52–54 | 只打乱训练集；验证集保持列表顺序，保证每次验证用的是同一批图。 |

### 3. `scripts/train_tadsr.sh`
| 行号 | 改动 |
|---|---|
| 14–18 | 新增 `--val_txt_path 'preset/val_list.txt'`、`--validation_steps 500`、`--val_timesteps 500`、`--best_metric psnr`、`--max_train_steps 100000`（原来没写，用的是默认值 100000）。 |

### 使用方法
1. 先为验证集生成列表（以 LOL-v2 Real 的测试集为例）：
   `python get_tag.py --ram_path preset/weights/ram_swin_large_14m.pth --lr_path <LOL-v2>/Real_captured/Test/Low --hr_path <LOL-v2>/Real_captured/Test/Normal --out_txt preset/val_list.txt`
2. `bash scripts/train_tadsr.sh`。训练到 `--max_train_steps` 会自动结束。
3. 输出在 `experience/TADSR/checkpoints/`：`model_best.pkl`（验证指标最好的权重）、`best.txt`（对应的步数和指标）、`model_*.pkl`（定期保存和最终权重）。

### 验证情况
- 改动过的 .py 通过了 `py_compile`。
- 用**真实的 `train_tadsr.py` 训练循环**在 CPU 上跑通（SD 模型换成很小的替身网络，数据是合成配对图）：
  - `--max_train_steps 7`（跨多个 epoch）时在第 7 步停止，保存了 `model_7.pkl`，进程正常退出。
  - 6 次验证中，后面 5 次指标都比第 1 次差时，`model_best.pkl` 和 `best.txt` 保持在第 1 次的结果，不会被覆盖。
  - 没给验证列表时打印 WARNING、不保存 best、训练正常结束；`--no-save_best` 时不做验证。
  - `--best_metric lpips`、梯度累积、`--val_num_images`、多个 `--val_timesteps`、尺寸不是 8 的倍数的验证图都正常。
  - 同一随机种子下，开和不开验证的训练 loss 序列完全一致，说明验证不干扰训练。
- **没有在 GPU 上用真实模型跑过**，也**没有测过多卡**。多卡时验证只在主进程上做，其他进程会在下一步同步梯度时等待。

---

## 2026-10-08 第 3 轮：time-aware skip（编码器→解码器旁路，保留文字和细节）

**目的**：SD 的 VAE 把图像压缩 8 倍、只留 4 个通道，小字和细纹理在 latent 里已经丢失，只改 latent 一侧救不回来。这一轮加了一条绕过 latent 的旁路：编码器 4 个分辨率的特征通过 1×1 卷积接到解码器对应层，并给解码器加 LoRA，和主模型**端到端一起训练**（思路同 CFW，但不需要第二阶段）。

**和 t 的关系**：旁路强度乘以 `gate(t)`，随 t 递减，所以 t 的语义不变并且更明确 —— 小 t 时旁路全开（忠实保留输入细节），大 t 时旁路减弱（更多交给生成先验）。默认 `gate(t) = sqrt(alpha_cumprod[t])`：t=1 时 1.00，t=250 时 0.82，t=500 时 0.53，t=750 时 0.24，t=999 时 0.07。

> 行号顺延提示：这一轮在以下文件里插入了代码，前两轮记录中的行号需要顺延。
> `tadsr.py`：整体下移较多，第 1 轮记的第 482–483 行现在在第 595–596 行；
> `train_tadsr.py`：第 137 行之后 +6；`test_tadsr.py`：第 66 行之后 +2。

### 1. `diffusers/models/autoencoders/vae.py`
| 行号 | 改动 |
|---|---|
| 292–295 | `TimeAwareEncoder.__init__`：新增 `store_skip_features`（开关）和 `current_skip_features`（存放特征）。 |
| 323–368 | `TimeAwareEncoder.forward`：开关打开时，记录每个 down block 的输入特征（分辨率 1、1/2、1/4、1/8；SD 配置下通道数 128、128、256、512）。三个分支（两种梯度检查点写法和普通写法）都加了。 |
| 478–483 | `Decoder.__init__`：新增 `skip_convs`、`incoming_skip_features`、`skip_gate` 三个属性。 |
| 485–505 | 新增 `Decoder.enable_skip_connections`：创建 4 个 1×1 卷积（SD 配置下 512→512、256→512、128→512、128→256），权重初始化为 1e-5。**在加载预训练权重之后**调用，避免 `from_pretrained` 因缺少权重报错。 |
| 507–521 | 新增 `Decoder._add_skip`：在每个 up block 之前加上 `skip_conv(编码器特征) × gate`。尺寸对不上时报错并提示输入必须是 8 的倍数。 |
| 552–553, 568–569, 577–578 | `Decoder.forward` 的三个分支里，在每个 up block 之前调用 `_add_skip`。 |

### 2. `tadsr.py`
| 行号 | 改动 |
|---|---|
| 18–71 | 新增 `SKIP_GATE_MODES`、`compute_skip_gate`（三种强度模式：`sqrt_alpha` / `linear` / `none`）、`get_vae_decoder_lora_modules`（解码器里加 LoRA 的 39 个模块）、`enable_vae_skip`（创建 skip 卷积 + 解码器 LoRA，并把编码器和解码器两个 LoRA **同时激活**）。 |
| 98–106 | `initialize_vae_time`：`use_skip` 时启用 skip，并把 LoRA、skip 卷积、time 参数设为可训练；返回值多了一项（解码器 LoRA 模块名）。 |
| 155–158 | `TADSR_gen.__init__`：记录 `use_skip` 和 `skip_gate_mode`，接收新的返回值。 |
| 215–221 | `load_pretrained_model`：预训练权重里带 skip 时一并加载。 |
| 235 | `set_train`：skip 卷积设为可训练。 |
| 280–290 | `TADSR_gen.forward`：解码前把编码器特征和 `gate(t)` 交给解码器，解码后立即清空。 |
| 298–303 | `save_model`：权重文件里新增 `use_skip`、`skip_gate`、`vae_lora_decoder_modules`、`state_dict_vae_skip`。 |
| 443–449 | `TADSR_test.__init__`：权重带 skip 时，**去掉编码器上的分块钩子**。原因：钩子让编码器始终走它自己的任务队列、不经过 `forward`，拿不到各层特征。代价是编码器不再分块，特别大的图更占显存。 |
| 482–497 | `TADSR_test.load_ckpt`：根据权重文件里的 `use_skip` 自动重建 skip 卷积和解码器 LoRA 并加载；没有这些键的旧权重按不带 skip 处理。 |
| 629–639 | `TADSR_test.forward`：解码前交接特征，强度为 `gate(t) × skip_scale`，解码后清空。 |

### 3. `train_tadsr.py`
| 行号 | 改动 |
|---|---|
| 139–143 | 新增 `--use_skip`（**默认开启**，`--no-use_skip` 关闭）和 `--skip_gate`（`sqrt_alpha` 默认 / `linear` / `none`）。 |

### 4. `test_tadsr.py`
| 行号 | 改动 |
|---|---|
| 67–68 | 新增 `--skip_scale`（默认 1.0）：推理时对旁路强度再乘一个系数，设 0 相当于关掉旁路，方便对比。是否使用 skip、用哪种 gate 由权重文件决定，不需要在命令行指定。 |

### 5. 脚本
| 文件 | 行号 | 改动 |
|---|---|---|
| `scripts/train_tadsr.sh` | 19–20 | 新增 `--use_skip`、`--skip_gate sqrt_alpha`。 |
| `scripts/test_tadsr.sh` | 13 | 新增 `--skip_scale 1.0`。 |
| `train_tadsr.py`、两个 sh | 文件末尾 | 去掉了前几轮转换换行符时留在文件最末尾的一个多余回车符。 |

### 对照实验怎么跑
- 不带旁路的基线：`--no-use_skip`
- 旁路强度不随 t 变化（普通 skip）：`--skip_gate none`
- 推理时临时关掉旁路看差别：`--skip_scale 0`

### 验证情况
用项目自带的 diffusers 0.25.0、peft 0.9.0、transformers 4.28.1（和 `requirements.txt` 一致）在 CPU 上测试。VAE 用的是**真实的 `TimeAwareAutoencoderKL` 类**（随机初始化），UNet 用同一个类的小尺寸版本，文本编码器用替身。
- 可训练参数分组正确：编码器 LoRA、解码器 LoRA、time 参数、4 个 skip 卷积，没有多余的。SD 通道配置下 skip 卷积形状为 512→512、256→512、128→512、128→256。
- 初始化时带 skip 和不带 skip 的输出几乎相同（最大差 < 1e-3），即训练从原模型的行为出发。
- 编码器 LoRA 和解码器 LoRA 都确实生效；梯度能传到四组参数。
- 旁路对输出的影响随 t 递减，和 gate 一致（`sqrt_alpha` 模式下 t=1 → 999，影响从 0.099 降到 0.002；`none` 模式不乘 gate）。
- **训练路径和推理路径输出完全一致**（保存 → 加载 → 推理，三种 gate 模式、t=1/500/999，差值为 0）。
- `--skip_scale 0` 和不带 skip 的解码结果完全一致。
- UNet 分块分支和 skip 一起能正常运行；不带 skip 键的旧格式权重能正常加载并走原来的路径；输入尺寸不是 8 的倍数时报清楚的错。
- 用真实的 `train_tadsr.py` 跑通了三种配置（默认开启 / `--no-use_skip` / `--skip_gate none`）：训练、验证、保存 best、到步数停止、再加载推理都正常；训练后 skip 卷积和 39 个解码器 LoRA 的权重都发生了更新。
- **没有验证的**：真实 SD 权重下的效果（文字和细节是否真的保住）、GPU / fp16 / 多卡、显存占用。测试用的是随机权重，只能说明代码接线正确，不能说明效果。

---

## 2026-10-09 第 4 轮：堵住旁路捷径（A：不走旁路的重建损失；B：旁路特征归一化）

**背景**：第 3 轮训练 25000 步后（`model_best.pkl`）看 LOL-v2 Real 00739：t=100 到 900 的文字和细节几乎一样（高频能量 0.022 vs 0.020），`--skip_scale 0` 时整图模糊、结构拖影，墙面偏灰。说明模型把内容几乎全部走旁路，latent 里信息很少，扩散部分没被用上；大 t 时还学会了放大编码器特征来抵消 gate。

- **A**：训练时把同一个 latent **关掉旁路再解码一次**（解码器 LoRA 共用），对这次输出也算和主重建损失同样形式的 L2 + LPIPS，乘 `--lambda_noskip`。latent 本身必须能还原完整图像，旁路只能在此基础上补细节。这相当于把 CFW 两阶段训练里"第一阶段保证 latent 好"的约束放回端到端训练中。代价：每步多一次解码器前向和反向。
- **B**：旁路特征进入 1×1 卷积前，先**逐通道在空间维度上标准化**（无可学习参数的 instance norm）。特征幅度固定，编码器无法放大特征抵消 gate，t 才真正控制旁路强度；每个通道的均值被去掉，**整体亮度和颜色无法再经旁路从暗图直接传过去**，只能由 latent 提供，旁路只负责结构和细节。

> 行号顺延提示：`vae.py` 第 483 行之后 +4、第 520 行之后再 +9；`tadsr.py` 第 3 轮记录的行号在第 60 行之后依次下移 1～17 行（越往后移得越多）；`train_tadsr.py` 第 143 行之后 +7、第 422 行之后再 +8。

### 1. `diffusers/models/autoencoders/vae.py`
| 行号 | 改动 |
|---|---|
| 484–485 | `Decoder.__init__`：新增 `skip_feature_norm` 属性（默认关闭）。 |
| 487–494 | `enable_skip_connections` 新增参数 `feature_norm`，并保存到 `self.skip_feature_norm`。 |
| 521–529 | `_add_skip`：`skip_feature_norm` 开启时，对旁路特征做逐通道标准化（float32 计算，eps=1e-5）。 |

### 2. `tadsr.py`
| 行号 | 改动 |
|---|---|
| 61–64 | `enable_vae_skip` 新增参数 `feature_norm`，传给解码器。 |
| 102–103 | `initialize_vae_time`：按 `--skip_feature_norm` 启用归一化。 |
| 160–162 | `TADSR_gen.__init__`：记录 `skip_feature_norm`，以及是否计算不走旁路的输出（`use_skip` 且 `lambda_noskip > 0`）。 |
| 298–304 | `TADSR_gen.forward`：**训练模式下**再做一次不走旁路的解码；返回值从 4 个变成 5 个，第 5 个是这张图（验证/推理时为 None）。 |
| 314 | `save_model`：权重文件新增 `skip_feature_norm`。 |
| 462–463 | 推理启动时打印的信息里加上 feature norm 状态。 |
| 499–503 | `TADSR_test.load_ckpt`：按权重里的 `skip_feature_norm` 重建；第 3 轮的权重没有这个键，按关闭处理，结果与第 3 轮代码完全一致。 |

### 3. `train_tadsr.py`
| 行号 | 改动 |
|---|---|
| 144–150 | 新增 `--skip_feature_norm`（**默认开启**，`--no-skip_feature_norm` 关闭）和 `--lambda_noskip`（默认 1.0，设 0 关闭 A）。 |
| 423 | 接收模型的第 5 个返回值。 |
| 430–436 | A 的损失：对不走旁路的输出算模糊 L2（权重 `lambda_l2`）+ LPIPS（权重 `lambda_lpips`），再乘 `lambda_noskip`，加到总损失里。 |
| 481–482 | 日志里新增 `loss_noskip`（TensorBoard 可看）。 |

### 4. `scripts/train_tadsr.sh`（在你改过的版本上修改）
| 行号 | 改动 |
|---|---|
| 12 | 输出目录改为 `experience/toLLIE_AB`，避免覆盖第 3 轮的结果。 |
| 21–22 | 新增 `--skip_feature_norm`、`--lambda_noskip 1.0`。 |

`test_tadsr.py` 和 `scripts/test_tadsr.sh` 不需要改：A 只在训练时起作用，B 的开关存在权重文件里，推理时自动读取。

### 对照实验
- 只用 B：`--lambda_noskip 0`
- 只用 A：`--no-skip_feature_norm`
- 第 3 轮的设置：`--no-skip_feature_norm --lambda_noskip 0`

### 验证情况
环境同第 3 轮（项目自带 diffusers 0.25.0、peft 0.9.0、transformers 4.28.1，CPU，真实 VAE 类 + 小尺寸 UNet，随机权重）。
- A：训练模式返回不走旁路的输出，eval 模式返回 None；只用这项损失时，梯度能传到 UNet LoRA、编码器 LoRA、解码器 LoRA，**传不到 skip 卷积**（符合设计）。
- B：把编码器旁路特征整体放大 10 倍，输出变化 1.8e-4（不开归一化时变化 2.0）；给所有旁路特征加一个常数偏移，输出变化 4.4e-6，即整体亮度/颜色不经旁路传递。
- 初始化时带旁路和不带旁路的输出最大差 2.6e-3，仍从原模型的行为出发。
- 旁路对输出的影响随 t 按 gate 递减（t=1 → 999：0.23 → 0.0075）。
- 保存 → 加载 → 推理，训练路径和推理路径输出差值为 0（t=1/500/999）。
- 第 3 轮的权重用新代码推理，结果与第 3 轮代码**逐像素一致**。
- 真实的 `train_tadsr.py` 跑通了四种配置（默认 / `--lambda_noskip 0` / `--no-skip_feature_norm` / `--no-use_skip`），`loss_noskip` 正常记录和下降，验证、保存 best、到步数停止都正常。
- **没有验证的**：真实 SD 权重下的效果、GPU/fp16/多卡下的显存（A 每步多一次解码器前后向，显存和时间都会增加）。

---

## 2026-10-10 第 5 轮：新增评测脚本 `eval_llie.py`

新文件，不改动已有代码。按文件名把结果和 GT 配对，计算：
- `psnr` / `ssim`：RGB、[0,1]、整图不裁边；SSIM 为 11×11 高斯窗（σ=1.5）、三通道平均。
- `psnr_gtmean` / `ssim_gtmean`：先把预测图整体缩放到和 GT 相同的平均亮度再算，排除整体亮度偏差的影响。
- `lpips`：默认 AlexNet（`--lpips_net vgg` 可切换），首次运行会下载 AlexNet 权重。
- `sat_ratio`：预测图平均饱和度 / GT 平均饱和度，<1 表示比 GT 发灰。
- `chroma_err`：YCbCr 色度通道（Cb、Cr）与 GT 的平均绝对误差（0～255 刻度）。

`--pred_dirs` 可给多个目录，终端输出一张汇总表；每个目录下写入逐张图的 `metrics.csv`；`--show 00739` 打印指定图片的结果。

用法：
```bash
python eval_llie.py --gt_dir /mnt/d/LOLv2/Real_captured/Test/Normal \
    --pred_dirs output/AB/t100_s1 output/AB/t500_s1 output/AB/t900_s1 output/AB/t500_s0 \
    --show 00739
```

验证：在合成图上与 scikit-image 的 `peak_signal_noise_ratio` / `structural_similarity`（gaussian_weights=True, sigma=1.5, use_sample_covariance=False）对比，差值 < 5e-5；半饱和度的测试图 `sat_ratio` = 0.61，整体变暗 20% 的测试图只有 GT-mean 版 PSNR 恢复、`sat_ratio` 保持 1.00，行为符合设计。LPIPS 部分在这边只用替身测过调用流程，真实数值以你环境里的 `lpips` 包为准。

---

## 2026-10-11 第 6 轮：颜色统计损失（纠正整体偏灰）+ 按 GT-mean PSNR 选 best

**依据**（第 4 轮权重在 LOL-v2 Real 测试集上的评测 + 数据集统计）：
- 结果整体偏亮 12%–18%（`bright_ratio` 1.10–1.18）。但训练集 GT 平均亮度 0.443、测试集 0.369，训练集本身亮约 20%，所以**偏亮是训练/测试数据分布差异**，不是模型问题 → 不加亮度约束，主要指标看 GT-mean PSNR/SSIM。
- 结果整体偏灰（`sat_ratio` 0.76–0.88）。训练集 GT 饱和度 0.267、测试集 0.226，训练集反而**更**饱和，所以**偏灰是模型造成的** → 加颜色统计损失。

> 行号顺延提示：`train_tadsr.py` 中第 4 轮记录的行号，第 135 行之后依次下移 2～80 行（越往后移得越多）。

### 1. `train_tadsr.py`
| 行号 | 改动 |
|---|---|
| 135–138 | `--best_metric` 新增选项 `psnr_gtmean`（先把预测整体缩放到 GT 的平均亮度再算 PSNR，与 `eval_llie.py` 一致）。 |
| 153–157 | 新增 `--lambda_color`（默认 1.0，设 0 关闭）。 |
| 186–224 | 新增 `COLOR_STATISTIC_SCALES`、`chroma_statistics`、`color_statistics_loss`：每张图计算 6 个全局统计量——Cb/Cr 均值（整体色偏）、Cb/Cr 标准差、平均色度幅度、HSV 平均饱和度（颜色浓度；饱和度项对"整体抬亮导致的发灰"也敏感）——与 GT 比较 L1，各项先除以各自的典型量级（色度类 0.05、饱和度 0.25）再平均。**只约束全局统计量、不做逐像素约束、不含亮度项、与 t 无关。** |
| 248–254, 276–280, 290–291 | `run_validation` 额外计算 GT-mean PSNR 和饱和度比 `sat_ratio`。 |
| 453–467 | 选 best 时 `psnr`/`psnr_gtmean` 越高越好；`best.txt`、TensorBoard（`val/psnr_gtmean`、`val/sat_ratio`）和终端输出都加上这两个指标。 |
| 500–506 | 颜色统计损失加到总损失：带旁路的输出 × `lambda_color`，不走旁路的输出 × `lambda_color × lambda_noskip`（保证 latent 这条路也学到正确的颜色）。 |
| 553–554 | 日志新增 `loss_color`。 |

### 2. `scripts/train_tadsr.sh`（在你改过的版本上修改，保留了 GPU 4,5）
| 行号 | 改动 |
|---|---|
| 12 | 输出目录改为 `experience/toLLIE_color`，不覆盖第 4 轮结果。 |
| 17 | `--best_metric psnr_gtmean`。 |
| 23 | 新增 `--lambda_color 1.0`。 |

推理不需要改：颜色损失只在训练时起作用。

### 对照实验
- 第 4 轮设置：`--lambda_color 0 --best_metric psnr`
- 颜色变得过饱和时：把 `--lambda_color` 调到 0.5 左右。

### 验证情况
- 颜色损失单元测试（合成图）：相同图像为 0；饱和度降到 90%/80%/50% 时损失为 0.24/0.47/1.20（与 LPIPS×2、L2 等现有损失量级相近）；RGB 整体加 0.1（抬亮发灰）时 0.056；蓝通道 ×0.9（色偏）时 0.099；**打乱像素位置时为 0**（只约束全局统计量）；从 70% 饱和度出发做梯度下降，饱和度比 0.78 → 1.00，损失 0.71 → 0.01。
- 用真实的 `train_tadsr.py` 在 CPU 上跑通三种配置（默认 + `--best_metric psnr_gtmean` / `--lambda_color 0` / `--no-use_skip`）：`loss_color` 正常记录和下降；验证输出新增的 `psnr_gtmean`、`sat_ratio`；按 `psnr_gtmean` 选 best 时，后一次更差会保留前一次的 best。
- 没有在真实 SD 权重和 GPU 上验证效果。

---

## 2026-10-12 第 7 轮：旁路按频率拆分——低频颜色通路（不受 t 控制）

**依据**：第 6 轮的颜色统计损失只把平均 `sat_ratio` 从 0.80 提到 0.85，739 黄墙仍偏灰；偏灰集中在少数场景（00769–00772、00763、00779、00780 等）。你的分析文档（`LOL-v2 低光增强色彩问题分析总结`）确认：739 黄墙的颜色**在输入里是有的**（B≈0、R/G≈3–4，只是灰阶很少），是网络把"暗图整体偏黄"当成光源色偏校正掉了。第 4 轮的 B 对旁路特征逐通道减均值，恰好把输入的颜色挡在了旁路外，颜色只能靠 latent 去猜，于是回归到平均色；全局统计损失约束不了这种局部的暗部饱和色。

**做法**：每层旁路特征拆成两部分：
- **低频（颜色/曝光）**：平均池化到约 原图/32 的分辨率再双线性放大回去，经单独的 1×1 卷积（`skip_convs_color`）接入解码器；**不归一化、不乘 gate(t)**，任何 t 下都把输入的颜色带过去。平均还能压低暗部色度的量化和噪声。
- **高频（结构/细节）**：原特征减去低频，继续第 4 轮的逐通道归一化 + gate(t)。t 对细节的调节、防止编码器抵消 gate 的机制不变。

语义上：**颜色和曝光忠于输入、与 t 无关，t 只控制细节的生成程度。** A（不走旁路的重建损失，两条通路都关）、颜色统计损失都保留。

> 行号顺延提示：`vae.py` 第 13 行之后 +2、第 495 行之后再 +12、第 520 行之后再 +22……；`tadsr.py` 第 3、4 轮记录的行号在第 60 行之后依次下移 2～14 行；`train_tadsr.py` 第 157 行之后 +7。

### 1. `diffusers/models/autoencoders/vae.py`
| 行号 | 改动 |
|---|---|
| 14, 21 | 新增 `import math`、`import torch.nn.functional as F`。 |
| 488–494 | `Decoder.__init__`：新增 `skip_color_path`、`skip_convs_color`、`color_pool_factor`（默认 32）、`color_scale`。 |
| 496–508 | `enable_skip_connections` 新增参数 `color_path`、`color_pool_factor`。 |
| 525–534 | `color_path` 开启时创建 4 个低频 1×1 卷积，通道对应关系与 `skip_convs` 相同，初始化 1e-5。参数名含 `skip_conv`，训练、保存、加载时和 `skip_convs` 一起处理，也不会被套上 LoRA。 |
| 536–545 | 新增 `_lowpass`：按层换算池化倍数（1/8 层 ×4、1/4 层 ×8、1/2 层 ×16、全分辨率层 ×32），`adaptive_avg_pool2d` 后双线性放大回原尺寸，任意 8 的倍数尺寸都可用。 |
| 557–564 | `_add_skip`：先拆出低频，经颜色卷积得到颜色项；高频 = 原特征 − 低频，进入原来的归一化 + gate。 |
| 578–581 | 颜色项乘 `color_scale`（不乘 gate）后加到解码器特征上。 |

### 2. `tadsr.py`
| 行号 | 改动 |
|---|---|
| 61–67 | `enable_vae_skip` 新增 `color_path`、`color_pool_factor`，传给解码器。 |
| 104–107 | `initialize_vae_time`：按 `--skip_color_path`、`--skip_color_pool` 启用。 |
| 167–169 | `TADSR_gen.__init__`：记录这两个设置。 |
| 298 | 训练前向：`color_scale = 1`（与 t 无关）。 |
| 323–324 | `save_model`：权重文件新增 `skip_color_path`、`skip_color_pool`。 |
| 473 | 推理启动信息加上 color path 状态。 |
| 511–517 | `TADSR_test.load_ckpt`：按权重里的设置重建颜色通路；之前的权重没有这些键，按关闭处理，结果与之前**逐像素一致**。 |
| 666–667 | 推理前向：`color_scale = skip_scale`，`--skip_scale 0` 时两条通路都关闭（与训练中不走旁路的情形一致）。 |

### 3. `train_tadsr.py`
| 行号 | 改动 |
|---|---|
| 158–164 | 新增 `--skip_color_path`（**默认开启**，`--no-skip_color_path` 关闭）、`--skip_color_pool`（默认 32）。 |

### 4. `scripts/train_tadsr.sh`（保留你改成的 Linux 换行和 GPU 4,5）
| 行号 | 改动 |
|---|---|
| 12 | 输出目录改为 `experience/toLLIE_colorpath`。 |
| 24–25 | 新增 `--skip_color_path`、`--skip_color_pool 32`。 |

推理脚本不用改，颜色通路的设置存在权重里。

### 对照实验
- 第 6 轮设置：`--no-skip_color_path`
- 低通更强/更弱：`--skip_color_pool 64` / `16`（越大颜色越平滑、越不带细节；太小则可能把细节也从颜色通路带过去）。

### 验证情况
环境同前（项目自带 diffusers 0.25.0、peft 0.9.0、transformers 4.28.1，CPU，真实 VAE 类 + 小尺寸 UNet，随机权重）。
- 颜色通路完全不受 gate 影响：gate 从 1.0 改为 0.068，颜色通路的贡献差值为 0；细节通路的贡献随 gate 降到约 15%。
- 颜色通路传递整体颜色、滤掉细节：对特征加逐通道常数偏移，输出变化 0.159；加同幅度的逐像素棋盘格，输出变化 < 1e-5。细节通路对逐通道偏移的响应为 4.5e-7（归一化去掉了，同第 4 轮）。
- A 的不走旁路解码不会给颜色卷积和细节卷积传梯度（两条通路都关）；主损失两者都有梯度。
- 初始化时与不带旁路的模型最大差 2.7e-3；保存 → 加载 → 推理差值为 0（t=1/500/999）；`--skip_scale 0` 与完全不带旁路的解码一致。
- 第 3 轮、第 4 轮格式的旧权重用新代码推理，结果与改动前**逐像素一致**。
- 真实的 `train_tadsr.py` 跑通了默认（颜色通路开）和 `--no-skip_color_path` 两种配置，训练、验证、保存 best、停止都正常，颜色卷积在训练中被更新。
- 没有在真实 SD 权重和 GPU 上验证效果。

---

## 2026-10-12 第 8 轮：新增最优 t 分析脚本 `analyze_oracle_t.py`

新文件，不改动已有代码。用途：在决定做"自动选 t"之前，先用数据判断有没有收益、该做哪种。

用同一批低光输入在多个 t 下的推理结果，对照 GT 计算：
1. **全局单一 t**：所有图用同一个 t 时，哪个 t 最好（PSNR、GT-mean PSNR、LPIPS）。
2. **逐图最优 t**（用 GT 选的上限）：比单一 t 好多少；最优 t 的分布；最优 t 与输入亮度、噪声、信噪比的 Spearman 相关。
3. **逐块最优 t 融合**（用 GT 选的上限）：图切成小块（默认 32px），每块取误差最小的 t 拼成融合图，比单一 t、逐图最优分别好多少；块的最优 t 与该块输入局部信噪比的相关，以及按局部信噪比分档的最优 t 分布。

判读：2 的收益明显 → 值得做全局自动 t；3 的收益明显大于 2 → 值得做空间自适应融合；都不明显 → 固定一个 t 即可。

噪声用 Immerkær 快速噪声估计；在纹理丰富处它也会把结构算进去，所以局部"噪声"更准确地说是"高频能量"，低光图里通常以噪声为主。

用法：
```bash
python analyze_oracle_t.py \
    --gt_dir /mnt/d/LOLv2/Real_captured/Test/Normal \
    --lq_dir /mnt/d/LOLv2/Real_captured/Test/Low \
    --pred 100=output/fifth/t100/skip1 250=output/fifth/t250/skip1 500=output/fifth/t500/skip1 \
           700=output/fifth/t700/skip1 900=output/fifth/t900/skip1 \
    --out_dir output/fifth/oracle_t
```
输出：`summary.txt`（终端汇总）、`per_image.csv`、`per_patch_bins.csv`。`--no_lpips` 可跳过 LPIPS 加速，`--patch` 改块大小。

验证：在合成数据上测试（3 张"干净"图 + 3 张右半边输入噪声很大的图，人为设定小 t 在干净区域更好、大 t 在噪声区域更好）：逐图最优 t 正确分为 3 张 t100、3 张 t900；逐块分析中局部信噪比最低的一档 100% 选 t900、其余各档 100% 选 t100；融合的收益（+5.7 dB）大于逐图最优（+3.7 dB），与设定一致。LPIPS 部分在这边用替身测试，真实数值以 `lpips` 包为准。

---

## 2026-10-13 第 9 轮：确定性推理 + 未训练 t 的提示

**依据**：第 8 轮的最优 t 分析中，82/100 张图的 GT-mean PSNR 随 t 来回跳动（例如 00690：27.3 / 29.0 / 26.9 / 28.9 / 27.2），t=250 与 t=100 的逐图差值中位数 0.56 dB、均值却只有 −0.16 dB，说明逐图差异以不规则波动为主。两个可能来源：(1) 推理从 VAE 后验**采样**且 `--seed` 从未被使用，大 t 时采样噪声被 1/√ᾱ_t 放大；(2) 测试用的 t（100/250/500/700/900）都不在训练用过的 50 个 t 里。本轮处理 (1)，并对 (2) 给出提示。

### 1. `tadsr.py`
| 行号 | 改动 |
|---|---|
| 288–290 | `TADSR_gen.forward`：训练模式照旧从后验**采样**；eval 模式（训练中的验证）改用**后验均值**，验证结果确定，并与推理默认一致。 |
| 472–475 | `TADSR_test.__init__`：新增 `latent_mode`（`mean` 默认 / `sample`）。 |
| 580–582 | `TADSR_test.forward`：按 `latent_mode` 取后验均值或采样。 |

### 2. `test_tadsr.py`
| 行号 | 改动 |
|---|---|
| 69–70 | 新增 `--latent_mode {mean,sample}`，默认 `mean`。 |
| 89–94 | 若 `--timesteps` 不在训练用过的 `np.linspace(1, 999, 50, dtype=int)` 中，打印提示和最近的训练值（如 100 → 102、250 → 245、500 → 510、700 → 693、900 → 897）。 |
| 165–167 | 每张图推理前 `torch.manual_seed(--seed)`（原来 `--seed` 只被解析、没有被使用）。`sample` 模式下结果可复现，且与图片处理顺序无关。 |

### 3. `scripts/test_tadsr.sh`
| 行号 | 改动 |
|---|---|
| 14 | 新增 `--latent_mode mean`（保留了你的其他参数）。 |

### 影响
- 推理默认结果会和之前略有不同（之前是采样）。100 张图平均指标受影响很小；单张图的比较从此可以复现。
- 想复现旧的行为：`--latent_mode sample`（现在配合 `--seed` 可复现）。

### 验证
在 CPU 上用真实类测试：`mean` 模式下不同随机种子输出完全一致（t=1/500/999）；`sample` 模式下同一种子输出完全一致、不同种子不同；训练中验证（eval 模式）的输出与推理默认逐像素一致，且不受随机种子影响；训练模式仍然采样。未训练 t 的提示逻辑：100、500 会提示（最近 102、510），102、510 不提示。

## 2026-10-13 第 10 轮：`analyze_oracle_t.py` 防止 GT / 低光目录写反

**依据**：output/fifth/oracle_t 的 summary 里 PSNR≈8 dB、LPIPS≈0.56、`lq_brightness`≈108（LOL-v2 低光图应在 10–30 左右），而网络输出平均亮度≈110，正常。说明这次分析把 `--gt_dir` 和 `--lq_dir` 写反了，结论无效，脚本当时也没有报错。

### 1. `analyze_oracle_t.py`
| 行号 | 改动 |
|---|---|
| 143–150 | 计算前取前 10 张图，比较 `--gt_dir` 与 `--lq_dir` 的平均亮度；低光目录不比 GT 暗时直接退出，并提示两个目录可能写反。 |

### 验证
用合成的亮/暗图测试：目录正确时正常运行；目录写反时退出并给出提示。

---

## 2026-10-13 第 11 轮：输入预提亮（暗图先提到正常曝光再送进 VAE 编码器）

**依据**：低光图直接送入 SD 的 VAE 编码器时，像素几乎都在 [-1,1] 的 -0.9 附近，暗部几个灰阶的颜色差异只有约 0.02；预训练编码器只见过正常曝光的图，要靠 rank=4 的 LoRA 学出 10～30 倍增益，暗部颜色容易丢（如 00739 左下角墙面）。第 3～7 轮在解码端的旁路和颜色改动对整体指标影响约 0.3 dB，00739 左下角的饱和度和色相在各轮之间几乎不变，说明问题在更上游。已确认原图里有这块墙的颜色信息，其他方法中也有基本正常的。

**做法**：送入编码器之前，用一个确定的、没有可学习参数的变换提亮输入（只作用于 LQ，GT 不动）：
- `gain`（默认）：每张图按平均亮度算一个增益 k = target / mean(Y)，限制在 [1, max_gain]。已经够亮的图不会被压暗。
- R、G、B 乘**同一个**比例，色相和通道比例不变。不用 gamma，因为 gamma 会让通道比例向 1 收缩，本身就会让颜色变灰。
- 高光：亮度超过 knee（0.8）的部分做平滑压缩而不是截断，比例同时限制在 1/max(R,G,B) 以内，不会因为单通道截断而改变色相。
- 增益按**整张图**计算：训练时在裁剪之前做，推理时在整图上做，两边一致。
- `fixed`：所有图用同一个增益，保留图与图之间的相对亮度，作为对照。

### 1. 新文件 `my_utils/llie_brighten.py`
`brighten_input(image_01, mode, target, max_gain, fixed_gain, knee)`，输入输出都是 [0,1]，支持 [C,H,W] / [B,C,H,W]，内部用 float32 计算。

### 2. `dataloaders/paired_dataset.py`
| 行号 | 改动 |
|---|---|
| 15 | 导入 `brighten_input`。 |
| 105–113 | `__getitem__`：LQ 转成张量后、裁剪之前，在整图上提亮。训练集和验证集都经过这里，所以训练中的验证也是提亮后的输入。参数从 args 读取，args 里没有时按 `none` 处理。 |

### 3. `tadsr.py`
| 行号 | 改动 |
|---|---|
| 16–29 | 导入 `brighten_input`；新增 `get_brighten_config`，可以从 args 或权重文件里读取设置，旧权重没有这些键时按 `none` 处理。 |
| 183–185 | `TADSR_gen.__init__`：记录提亮设置。`TADSR_gen.forward` **不再提亮**，因为数据集已经处理过，避免重复提亮。 |
| 343–348 | `save_model`：权重文件新增 `input_brighten`、`input_brighten_target`、`input_brighten_max_gain`、`input_brighten_fixed_gain`、`input_brighten_knee`。 |
| 502 | 推理启动时打印提亮设置。 |
| 543–544 | `TADSR_test.load_ckpt`：从权重读取提亮设置。 |
| 605–608 | `TADSR_test.forward`：编码前在整图上提亮（float32），设置来自权重文件。之前的权重按 `none` 处理，结果与改动前一致。 |

### 4. `train_tadsr.py`
| 行号 | 改动 |
|---|---|
| 165–175 | 新增 `--input_brighten {none,gain,fixed}`（默认 `gain`）、`--input_brighten_target`（0.4）、`--input_brighten_max_gain`（20）、`--input_brighten_fixed_gain`（8）、`--input_brighten_knee`（0.8）。 |

### 5. `scripts/train_tadsr.sh`
| 行号 | 改动 |
|---|---|
| 12 | 输出目录改为 `experience/toLLIE_brighten`。 |
| 26–29 | 新增 `--input_brighten gain`、`--input_brighten_target 0.4`、`--input_brighten_max_gain 20`、`--input_brighten_knee 0.8`。 |

`test_tadsr.py` 和推理脚本不用改：提亮设置存在权重里，推理时自动按同样的方式处理整图。RAM 打 tag 用的 `--ram_brighten` 是另一套（gamma，只影响 tag），保持不变。

### 对照实验
- 不提亮（第 7 轮设置）：`--input_brighten none`
- 固定增益：`--input_brighten fixed --input_brighten_fixed_gain 8`
- 目标亮度：`--input_brighten_target 0.3 / 0.5`

### 验证情况
- 改动过的 .py 都通过了语法检查（`ast.parse`）。
- **没有运行任何测试**（按你的要求，不在本地或云端用 CPU 跑），需要在服务器上确认：训练能正常启动；第一次验证的指标量级正常；推理时打印的 `input brighten` 设置和训练时一致。

---

## 2026-10-13 第 12 轮：后台训练脚本（SSH 断开后继续运行）

新增两个脚本，`scripts/train_tadsr.sh` 不变。
- `scripts/train_bg.sh [训练脚本]`：默认运行 `scripts/train_tadsr.sh`。用 `setsid` + `nohup` 启动，脱离终端，SSH 断开不受影响。输出写到 `logs/train_时间.log`，`logs/train_latest.log` 指向最新的日志，PID 写到 `logs/train.pid`。已有训练在运行时拒绝重复启动；进程启动后立即退出时打印日志末尾。
- `scripts/stop_train.sh`：向整个进程组发送 TERM，连同 accelerate 启动的各卡子进程一起停掉；30 秒没退出就强制结束。

验证：用一个带两个子进程的假训练脚本测试（不涉及真实训练）。启动后三个进程在同一个新会话和进程组里；重复启动被拒绝；停止后进程组里没有残留进程，日志正常写入。
