import os
import glob
import openslide
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ================= 配置区域 (Config) =================
# 1. 输入：解压后的 GDC 数据根目录 (脚本会自动递归搜索里面的 .svs)
# 假设客户解压到了 raw_data 文件夹
SOURCE_ROOT = "/mnt/MR-PLIP/raw_data" 

# 2. 输出：切片保存路径
OUTPUT_ROOT = "/mnt/MR-PLIP/dataset/gdc_processed_images"
# 跑一次更新一次图片

# 3. 并发线程数 (服务器一般核多，设为 8 或 16 都可以)
NUM_THREADS = 8

# [cite_start]4. 论文定义的倍率层级 [cite: 246-248]
# 假设 SVS 的 Base Level (Level 0) 是 40x (这是 TCGA 的标准)
# Downsample: 相对于 40x 的缩小倍数
HIERARCHY = {
    '5x':  {'downsample': 8, 'count': 1},   # 40x / 8 = 5x
    '10x': {'downsample': 4, 'count': 4},   # 40x / 4 = 10x
    '20x': {'downsample': 2, 'count': 16},  # 40x / 2 = 20x
    '40x': {'downsample': 1, 'count': 64}   # 原生分辨率
}
PATCH_SIZE = 224
# ====================================================

def get_tissue_mask(slide, level_idx):
    """
    [Pro功能] 智能去背景 (Otsu Thresholding)
    计算载玻片上有组织的区域，防止切到空白玻璃。
    """
    try:
        # 读取指定 Level 的缩略图
        thumb = slide.read_region((0,0), level_idx, slide.level_dimensions[level_idx])
        thumb = np.array(thumb.convert('RGB'))
        
        # 转灰度 -> Otsu 二值化
        gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask, thumb.shape
    except:
        return None, None

def process_one_slide(svs_path):
    # 提取 Slide ID (文件名去掉后缀)
    slide_id = os.path.basename(svs_path).replace('.svs', '')
    save_dir = os.path.join(OUTPUT_ROOT, slide_id)
    
    # 断点续传：如果已经切完了，跳过
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 85:
        print(f"✅ Skipping {slide_id} (Already processed)")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f">>> [Processing] {slide_id} ...")

    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception as e:
        print(f"❌ Error opening {svs_path}: {e}")
        return

    # 获取最高分辨率 (Level 0) 尺寸
    w, h = slide.dimensions
    
    # 简单策略：Center Crop (取中心区域)
    # 工业界更复杂的做法是基于 Mask 随机采样，但为了复现论文结构，
    # 我们以中心为原点，向外金字塔式扩散。
    center_x, center_y = w // 2, h // 2
    
    # 遍历 4 个层级
    for res_name, config in HIERARCHY.items():
        count = config['count']
        ds = config['downsample']
        
        # 计算该倍率下，需要的实际物理区域大小 (在 Level 0 上)
        # 例如 5x 的一张图 (224px)，相当于在 40x 上切 224*8 = 1792px
        region_w = PATCH_SIZE * ds
        
        # 计算网格布局 (Grid Layout)
        # 1 -> 1x1, 4 -> 2x2, 16 -> 4x4, 64 -> 8x8
        grid_side = int(np.sqrt(count))
        
        # 计算起始偏移量 (Top-Left)，保证 Grid 居中
        start_x = center_x - (grid_side * region_w) // 2
        start_y = center_y - (grid_side * region_w) // 2
        
        saved_count = 0
        for row in range(grid_side):
            for col in range(grid_side):
                # 计算当前 Patch 在 Level 0 的坐标
                x = start_x + col * region_w
                y = start_y + row * region_w
                
                try:
                    # 核心动作：切图 -> 缩放
                    # read_region(location, level, size) 这里的 level 0 是最高清
                    patch = slide.read_region((x, y), 0, (region_w, region_w))
                    patch = patch.convert("RGB")
                    
                    # 如果需要缩放 (比如 5x 需要把 1792px 缩放到 224px)
                    if ds != 1:
                        patch = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BICUBIC)
                    
                    # 命名格式: 5x.png 或 10x_3.png
                    if count == 1:
                        fname = f"{res_name}.png"
                    else:
                        fname = f"{res_name}_{saved_count}.png"
                        
                    patch.save(os.path.join(save_dir, fname))
                    saved_count += 1
                    
                except Exception as e:
                    print(f"    Warning: Failed to cut {res_name} patch {saved_count}: {e}")

    print(f"✅ Slide {slide_id} Done. Saved to {save_dir}")

def main():
    # 递归搜索所有 .svs 文件 (穿透 UUID 文件夹)
    # raw_data/**/*.svs
    search_pattern = os.path.join(SOURCE_ROOT, "**", "*.svs")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"❌ No .svs files found in {SOURCE_ROOT}. Please check path.")
        return
    
    print(f"🎯 Found {len(files)} slides. Starting Multi-threaded Patcher...")
    print(f"   Threads: {NUM_THREADS}")
    
    # 线程池并发处理
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(process_one_slide, files)
        
    print("\n🎉 All slides processed.")

if __name__ == "__main__":
    from PIL import Image # Lazy import
    main()