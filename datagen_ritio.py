import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
from scipy import ndimage

warnings.filterwarnings('ignore')


def load_all_csv_data(csv_folder_path):
    """加载所有CSV文件并合并成一个长序列"""
    print("正在加载和合并所有CSV文件...")

    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.xlsx')]
    csv_files.sort()

    all_data_sequences = []

    for csv_file in tqdm(csv_files, desc="读取文件"):
        file_path = os.path.join(csv_folder_path, csv_file)
        try:
            df = pd.read_excel(file_path)

            # 根据实际数据结构调整这里
            if 'value' in df.columns:
                data = df['value'].values
            elif '日照时数' in df.columns:
                data = df[
                    '日照时数'].values  # 最高温度(℃)	最低温度(℃)	平均温度(℃)	平均水汽压(hPa)	平均相对湿度(%RH)	日降水	风速(最大)	平均风速(10m)	日照时数
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data = df[numeric_cols[0]].values
                else:
                    continue

            all_data_sequences.append(data)

        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {e}")

    combined_data = np.concatenate(all_data_sequences)
    print(f"合并后的数据总长度: {len(combined_data)}")

    return combined_data


def normalize_data(data_sequence):
    """对数据进行归一化处理"""
    print("正在进行数据归一化...")

    valid_data = data_sequence[~np.isnan(data_sequence)]

    if len(valid_data) == 0:
        raise ValueError("没有有效的数值数据进行归一化")

    scaler = StandardScaler()
    valid_data_2d = valid_data.reshape(-1, 1)
    scaler.fit(valid_data_2d)

    data_2d = data_sequence.reshape(-1, 1)
    normalized_data = scaler.transform(data_2d).flatten()

    print(f"归一化参数 - 均值: {scaler.mean_[0]:.4f}, 标准差: {scaler.scale_[0]:.4f}")

    return normalized_data, scaler


def create_sliding_windows_fixed(data_sequence, window_size=105, stride=105):
    """使用固定间隔的滑动窗口生成图像"""
    print("正在生成固定间隔滑动窗口图像...")

    windows = []
    total_length = len(data_sequence)

    num_windows = (total_length - window_size * window_size) // (stride * window_size) + 1
    print(f"预计生成窗口数量: {num_windows}")

    for i in tqdm(range(num_windows), desc="生成窗口"):
        start_idx = i * stride * window_size
        end_idx = start_idx + window_size * window_size

        if end_idx > total_length:
            break

        window_data = data_sequence[start_idx:end_idx]
        image = window_data.reshape(window_size, window_size)
        windows.append(image)

    data_matrix = np.array(windows)
    print(f"实际生成的图像数量: {data_matrix.shape[0]}")

    return data_matrix


def create_datasets_with_nan_zero(data_matrix):
    """创建数据集，将有NaN的图像中的NaN值置零"""
    print("正在创建数据集并将NaN值置零...")

    has_nan = np.array([np.any(np.isnan(img)) for img in data_matrix])
    no_nan = ~has_nan

    print(f"包含NaN的图像数量: {np.sum(has_nan)}")
    print(f"不包含NaN的图像数量: {np.sum(no_nan)}")

    # 处理有NaN的图像：将NaN置零
    completion_data = data_matrix[has_nan].copy()
    for i in range(len(completion_data)):
        nan_mask = np.isnan(completion_data[i])
        completion_data[i][nan_mask] = 0.0

    # 完整数据集（不包含NaN）
    complete_data = data_matrix[no_nan]

    # 分割完整数据为训练集和验证集
    if len(complete_data) > 1:
        train_data, val_data = train_test_split(
            complete_data, test_size=0.2, random_state=42, shuffle=True
        )
    else:
        train_data = complete_data
        val_data = complete_data
        print("警告：完整数据太少，无法正常分割")

    print(f"训练集大小: {train_data.shape}")
    print(f"验证集大小: {val_data.shape}")
    print(f"待补齐集大小: {completion_data.shape}")

    return train_data, val_data, completion_data


def apply_realistic_mask_with_ratio(data, target_mask_ratio=0.3,
                                    min_stations_per_group=3, max_stations_per_group=10,
                                    min_time_length=10, max_time_length=30,
                                    max_iterations=100):
    """
    应用更真实的掩膜模式，并控制掩膜比例

    Parameters:
        data: 输入数据 [time_length, num_stations]
        target_mask_ratio: 目标掩膜比例 (0-1)
        min_stations_per_group: 每组最小站点数
        max_stations_per_group: 每组最大站点数
        min_time_length: 最小时间长度
        max_time_length: 最大时间长度
        max_iterations: 最大迭代次数
    """
    masked_data = data.copy()
    mask = np.ones_like(data)  # 1表示已知，0表示缺失

    time_length, num_stations = data.shape
    current_mask_ratio = 0.0
    iteration = 0

    # 计算目标掩膜点数
    total_points = time_length * num_stations
    target_masked_points = int(total_points * target_mask_ratio)

    print(f"目标掩膜比例: {target_mask_ratio:.2%}, 目标掩膜点数: {target_masked_points}")

    while current_mask_ratio < target_mask_ratio and iteration < max_iterations:
        # 随机选择一组站点
        stations_in_group = np.random.randint(min_stations_per_group, max_stations_per_group + 1)
        station_indices = np.random.choice(num_stations, size=stations_in_group, replace=False)

        # 随机选择连续的时间段
        time_length_group = np.random.randint(min_time_length, max_time_length + 1)
        time_start = np.random.randint(0, time_length - time_length_group)
        time_end = time_start + time_length_group

        # 检查这些位置是否已经被掩膜
        current_region_mask = mask[time_start:time_end, station_indices]
        already_masked_points = np.sum(current_region_mask == 0)
        new_masked_points = len(station_indices) * time_length_group - already_masked_points

        # 如果添加这个掩膜不会超过目标比例太多，则应用掩膜
        if (current_mask_ratio * total_points + new_masked_points) <= target_masked_points * 1.1:
            # 应用掩膜：这组站点在这段时间内全部缺失
            masked_data[time_start:time_end, station_indices] = 0
            mask[time_start:time_end, station_indices] = 0

            current_mask_ratio = 1 - np.mean(mask)
            print(f"  迭代 {iteration}: 添加掩膜区域 - {new_masked_points}点, 当前比例: {current_mask_ratio:.2%}")

        iteration += 1

        # 如果接近目标比例，提前退出
        if abs(current_mask_ratio - target_mask_ratio) < 0.02:
            break

    # 如果掩膜比例不足，随机添加一些点
    if current_mask_ratio < target_mask_ratio * 0.9:
        remaining_points = target_masked_points - int(current_mask_ratio * total_points)
        if remaining_points > 0:
            # 找到所有未被掩膜的位置
            unmasked_positions = np.where(mask == 1)
            if len(unmasked_positions[0]) > 0:
                # 随机选择一些位置进行掩膜
                selected_indices = np.random.choice(len(unmasked_positions[0]),
                                                    size=min(remaining_points, len(unmasked_positions[0])),
                                                    replace=False)

                for idx in selected_indices:
                    i, j = unmasked_positions[0][idx], unmasked_positions[1][idx]
                    masked_data[i, j] = 0
                    mask[i, j] = 0

                current_mask_ratio = 1 - np.mean(mask)
                print(f"  补充随机掩膜: {len(selected_indices)}点, 最终比例: {current_mask_ratio:.2%}")

    final_mask_ratio = 1 - np.mean(mask)
    print(f"最终掩膜比例: {final_mask_ratio:.2%} (目标: {target_mask_ratio:.2%})")

    return masked_data, mask


def apply_masks_to_datasets_realistic(train_data, val_data,
                                      target_mask_ratio=0.3,
                                      min_stations_per_group=3, max_stations_per_group=10,
                                      min_time_length=10, max_time_length=30):
    """
    对训练集和验证集应用更真实的掩膜处理

    Parameters:
        target_mask_ratio: 目标掩膜比例 (0-1)
    """
    print("正在应用真实掩膜模式...")
    print(f"目标掩膜比例: {target_mask_ratio:.2%}")
    print(f"掩膜参数: 每组{min_stations_per_group}-{max_stations_per_group}个站点")
    print(f"缺失时长: {min_time_length}-{max_time_length}个时间步")

    masked_train_data = []
    train_masks = []
    masked_val_data = []
    val_masks = []

    # 对训练集应用掩膜
    for i in tqdm(range(len(train_data)), desc="掩膜训练集"):
        original_img = train_data[i]
        masked_img, mask = apply_realistic_mask_with_ratio(
            original_img, target_mask_ratio,
            min_stations_per_group, max_stations_per_group,
            min_time_length, max_time_length
        )
        masked_train_data.append(masked_img)
        train_masks.append(mask)

    # 对验证集应用掩膜
    for i in tqdm(range(len(val_data)), desc="掩膜验证集"):
        original_img = val_data[i]
        masked_img, mask = apply_realistic_mask_with_ratio(
            original_img, target_mask_ratio,
            min_stations_per_group, max_stations_per_group,
            min_time_length, max_time_length
        )
        masked_val_data.append(masked_img)
        val_masks.append(mask)

    masked_train_data = np.array(masked_train_data)
    train_masks = np.array(train_masks)
    masked_val_data = np.array(masked_val_data)
    val_masks = np.array(val_masks)

    actual_train_ratio = np.mean(1 - train_masks)
    actual_val_ratio = np.mean(1 - val_masks)

    print(f"训练集实际掩膜比例: {actual_train_ratio:.4f} (目标: {target_mask_ratio:.4f})")
    print(f"验证集实际掩膜比例: {actual_val_ratio:.4f} (目标: {target_mask_ratio:.4f})")

    return masked_train_data, train_masks, masked_val_data, val_masks


def analyze_mask_pattern(mask):
    """
    分析掩膜模式，识别连续的缺失块
    """
    labeled_mask, num_features = ndimage.label(mask == 0)

    print(f"识别到 {num_features} 个连续缺失区域:")

    for i in range(1, num_features + 1):
        region_mask = labeled_mask == i
        if np.any(region_mask):
            time_indices, station_indices = np.where(region_mask)
            time_span = time_indices.max() - time_indices.min() + 1
            station_span = station_indices.max() - station_indices.min() + 1

            print(f"  区域 {i}: {len(time_indices)}个点, "
                  f"时间范围[{time_indices.min()}-{time_indices.max()}]({time_span}步), "
                  f"站点范围[{station_indices.min()}-{station_indices.max()}]({station_span}站)")


def visualize_realistic_masks(train_data, masked_train_data, train_masks,
                              val_data, masked_val_data, val_masks,
                              completion_dataset, target_mask_ratio):
    """
    可视化真实掩膜模式
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'气象数据可视化 - 真实掩膜模式 (目标掩膜比例: {target_mask_ratio:.1%})', fontsize=16, fontweight='bold')

    # 训练集可视化
    if len(train_data) >= 1:
        # 原始样本
        img_original = train_data[0]
        im0 = axes[0, 0].imshow(img_original, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'训练集原始样本\n形状: {img_original.shape}')
        axes[0, 0].set_xlabel('站点维度')
        axes[0, 0].set_ylabel('时间维度')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 掩膜样本
        img_masked = masked_train_data[0]
        mask = train_masks[0]

        display_img = img_original.copy()
        mask_regions = mask == 0
        display_img[mask_regions] = np.nan

        im1 = axes[0, 1].imshow(display_img, cmap='viridis', aspect='auto')
        axes[0, 1].imshow(np.where(mask_regions, 1, np.nan), cmap='Reds', alpha=0.6, aspect='auto')

        masked_ratio = 1 - np.mean(mask)
        axes[0, 1].set_title(f'训练集掩膜样本\n实际掩膜比例: {masked_ratio:.2%}\n红色区域: 模拟缺失')
        axes[0, 1].set_xlabel('站点维度')
        axes[0, 1].set_ylabel('时间维度')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 验证集可视化
    if len(val_data) >= 1:
        img_original = val_data[0]
        im0 = axes[1, 0].imshow(img_original, cmap='plasma', aspect='auto')
        axes[1, 0].set_title(f'验证集原始样本\n形状: {img_original.shape}')
        axes[1, 0].set_xlabel('站点维度')
        axes[1, 0].set_ylabel('时间维度')
        plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

        img_masked = masked_val_data[0]
        mask = val_masks[0]

        display_img = img_original.copy()
        mask_regions = mask == 0
        display_img[mask_regions] = np.nan

        im1 = axes[1, 1].imshow(display_img, cmap='plasma', aspect='auto')
        axes[1, 1].imshow(np.where(mask_regions, 1, np.nan), cmap='Reds', alpha=0.6, aspect='auto')

        masked_ratio = 1 - np.mean(mask)
        axes[1, 1].set_title(f'验证集掩膜样本\n实际掩膜比例: {masked_ratio:.2%}\n红色区域: 模拟缺失')
        axes[1, 1].set_xlabel('站点维度')
        axes[1, 1].set_ylabel('时间维度')
        plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 待补齐集可视化
    if len(completion_dataset) >= 1:
        img = completion_dataset[0]
        original_nan_mask = img == 0

        display_img = img.copy()
        display_img[original_nan_mask] = np.nan

        im2 = axes[2, 0].imshow(display_img, cmap='coolwarm', aspect='auto')
        axes[2, 0].imshow(np.where(original_nan_mask, 1, np.nan), cmap='Blues', alpha=0.6, aspect='auto')

        zero_ratio = np.mean(original_nan_mask)
        axes[2, 0].set_title(f'待补齐集样本\n原始NaN比例: {zero_ratio:.2%}\n蓝色区域: 原始缺失')
        axes[2, 0].set_xlabel('站点维度')
        axes[2, 0].set_ylabel('时间维度')
        plt.colorbar(im2, ax=axes[2, 0], fraction=0.046, pad=0.04)

        # 显示统计信息
        axes[2, 1].axis('off')
        stats_text = f"数据集统计:\n"
        stats_text += f"训练集: {train_data.shape}\n"
        stats_text += f"验证集: {val_data.shape}\n"
        stats_text += f"待补齐集: {completion_dataset.shape}\n"
        stats_text += f"目标掩膜比例: {target_mask_ratio:.2%}\n"
        stats_text += f"实际掩膜比例: {np.mean(1 - train_masks):.2%}"
        axes[2, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                        transform=axes[2, 1].transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()
    plt.show()


def save_masked_datasets(train_data, masked_train_data, train_masks,
                         val_data, masked_val_data, val_masks,
                         completion_dataset, scaler, output_folder='./weather_masked_datasets'):
    """保存掩膜后的数据集"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存原始数据
    np.save(os.path.join(output_folder, 'train_original.npy'), train_data)
    np.save(os.path.join(output_folder, 'val_original.npy'), val_data)

    # 保存掩膜数据
    np.save(os.path.join(output_folder, 'train_masked.npy'), masked_train_data)
    np.save(os.path.join(output_folder, 'train_masks.npy'), train_masks)
    np.save(os.path.join(output_folder, 'val_masked.npy'), masked_val_data)
    np.save(os.path.join(output_folder, 'val_masks.npy'), val_masks)

    # 保存待补齐集
    np.save(os.path.join(output_folder, 'completion_dataset.npy'), completion_dataset)

    # 保存归一化参数
    np.save(os.path.join(output_folder, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(output_folder, 'scaler_scale.npy'), scaler.scale_)

    stats = {
        'train_original_shape': train_data.shape,
        'train_masked_shape': masked_train_data.shape,
        'val_original_shape': val_data.shape,
        'val_masked_shape': masked_val_data.shape,
        'completion_shape': completion_dataset.shape,
        'train_masked_ratio': np.mean(1 - train_masks),
        'val_masked_ratio': np.mean(1 - val_masks),
        'completion_zero_ratio': np.mean(completion_dataset == 0),
        'normalization_mean': scaler.mean_[0],
        'normalization_std': scaler.scale_[0]
    }

    with open(os.path.join(output_folder, 'dataset_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    print(f"掩膜数据集已保存到: {output_folder}")


def generate_synthetic_long_sequence_normalized():
    """生成长序列合成数据并归一化"""
    print("生成合成长序列气象数据...")

    np.random.seed(42)
    total_length = 50000
    time_axis = np.linspace(0, 20 * np.pi, total_length)

    base_sequence = (np.sin(time_axis) +
                     0.5 * np.sin(2 * time_axis) +
                     0.2 * np.sin(0.5 * time_axis) +
                     0.1 * time_axis / (2 * np.pi) +
                     0.3 * np.random.randn(total_length))

    data_with_nan = base_sequence.copy()
    nan_probability = 0.02

    nan_positions = np.random.choice(total_length, size=int(total_length * nan_probability), replace=False)
    data_with_nan[nan_positions] = np.nan

    for _ in range(20):
        block_start = np.random.randint(0, total_length - 50)
        block_length = np.random.randint(10, 50)
        data_with_nan[block_start:block_start + block_length] = np.nan

    return data_with_nan


def main():
    csv_folder_path = "I:/datasets/1961to2022xinjiangclimates数据补齐"

    # 在这里设置目标掩膜比例
    target_mask_ratio = 0.9# 30%的掩膜比例，可以根据需要修改

    try:
        data_sequence = load_all_csv_data(csv_folder_path)
    except Exception as e:
        print(f"加载真实数据失败: {e}")
        data_sequence = generate_synthetic_long_sequence_normalized()

    normalized_data, scaler = normalize_data(data_sequence)
    data_matrix = create_sliding_windows_fixed(normalized_data, window_size=105, stride=105)
    train_data, val_data, completion_dataset = create_datasets_with_nan_zero(data_matrix)

    # 应用真实掩膜
    masked_train_data, train_masks, masked_val_data, val_masks = apply_masks_to_datasets_realistic(
        train_data, val_data,
        target_mask_ratio=target_mask_ratio,
        min_stations_per_group=3,
        max_stations_per_group=10,
        min_time_length=10,
        max_time_length=30
    )

    # 分析掩膜模式
    if len(train_masks) > 0:
        print("\n=== 训练集掩膜模式分析 ===")
        analyze_mask_pattern(train_masks[0])

    visualize_realistic_masks(
        train_data, masked_train_data, train_masks,
        val_data, masked_val_data, val_masks,
        completion_dataset, target_mask_ratio
    )

    save_masked_datasets(
        train_data, masked_train_data, train_masks,
        val_data, masked_val_data, val_masks,
        completion_dataset, scaler
    )

    print("\n=== 数据集统计 ===")
    print(f"总样本数: {len(train_data) + len(val_data) + len(completion_dataset)}")
    print(f"目标掩膜比例: {target_mask_ratio:.2%}")
    print(f"实际掩膜比例: {np.mean(1 - train_masks):.2%}")


if __name__ == "__main__":
    main()