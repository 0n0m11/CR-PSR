# CR-WSI-VQA

Cross-resolution Pseudo-semantic Representation for Multi-magnification Laryngeal H&E Pathology VQA（跨分辨率伪语义表示 · 多倍率喉癌病理 VQA）。

> 本仓库当前上传 **S5 伪语义描述符管线**（论文核心贡献之一），其余模块（S1–S4 对齐、S6 报告、S7 数据、S8 VQA 训练、评估工具）将随后续批次补充。

---

## 一、总体框架

![CR-WSI-VQA 总体框架](fig1.png)

---

## 二、S5 伪语义描述符管线

S5 是"跨分辨率伪语义表示"的核心：用冻结的 Quilt-LLaVA 生成结构化伪标签，再训练一个紧凑的描述符模型，产出可检查的中间表示（20× 结构化描述符 + 软证据）。

| 脚本 | 作用 |
|------|------|
| `S5a_make_tag_schema_20x40x_shrunk.py` | 定义 20×/40× 描述符标签 schema |
| `S5a0_build_patch_desc_dataset.py` | 构建 patch 描述符数据集 |
| `S5b_build_patch_table_20x40x.py` | 构建 20×/40× patch 表 |
| `S5c_bootstrap_pseudolabels_quilt.py` | Quilt-LLaVA 伪标签 bootstrap（冻结模型） |
| `S5d_stabilize_pseudolabels.py` | 伪标签稳定化 |
| `S5e_train_desc_child_only.py` | 描述符训练（child-only） |
| `S5f_train_desc_fusion.py` | 描述符训练（fusion） |
| `S5g_infer_desc_hybrid.py` | 描述符推理（hybrid） |
| `S5h_infer_desc_child_only.py` | 描述符推理（child-only） |
| `S5i_export_20x_soft_scores.py` | 导出 20× soft 得分 |
| `S5j_merge_multiscale_pseudocaption.py` | 多倍率描述符合并 |
| `modeling_hierapatch_desc.py` | 描述符模型定义 |
| `verbalizer.py` | 描述符 → 文本转写 |

---

## 三、目录结构

```
CR-WSI-VQA/
├── README.md
├── fig1.png             # 总体框架图
├── common.py            # 公共模块（S5 共用工具：load/save/split/subset）
└── s5_descriptor/       # S5 伪语义描述符管线（13 文件）
```

---

## 四、环境

- 环境：`llava_env`（S5，Python 3.10，torch 2.1.2+cu121）
- 依赖：见仓库根目录 `requirements.txt`（随后续批次补充）

**运行时环境变量**：

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/llava_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/root/miniconda3/envs/mr_train/lib:/root/miniconda3/envs/llava_env/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/CR-WSI-VQA
```

> `PYTHONPATH` 指向仓库根目录，保证 `s5_descriptor/` 脚本的 `from common import ...` 能解析到顶层 `common.py`。

---

## 五、数据与权重（未纳入仓库）

| 项 | 说明 |
|----|------|
| Quilt-LLaVA | `checkpoints/Quilt-Llava-v1.5-7b/`（S5c 依赖，需从 HF 下载） |
| 切图 | `dataset/mrplip_processed/`（5×/10×/20×/40×） |

---

## 六、复现入口

```bash
cd /path/to/CR-WSI-VQA
export PYTHONPATH=/path/to/CR-WSI-VQA
export LD_LIBRARY_PATH=...（见上）

# S5 伪标签 bootstrap
python s5_descriptor/S5c_bootstrap_pseudolabels_quilt.py --help

# S5 描述符训练
python s5_descriptor/S5e_train_desc_child_only.py --help
```

---

## 七、后续批次（待补充）

- S1–S4 跨分辨率对齐
- S6 报告生成
- S7 VQA 数据构建
- S8 VQA 训练（主结果 66.21% acc / 43.40% macro-F1）
- 评估工具与敏感性分析
