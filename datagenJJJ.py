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

    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    csv_files.sort()

    all_data_sequences = []

    for csv_file in tqdm(csv_files, desc="读取文件"):
        file_path = os.path.join(csv_folder_path, csv_file)
        try:
            df = pd.read_csv(file_path)

            # 根据实际数据结构调整这里
            if 'value' in df.columns:
                data = df['value'].values
            elif '平均风速(10m)' in df.columns:
                data = df['平均风速(10m)'].values  # 最高温度(℃)	最低温度(℃)	平均温度(℃)	平均水汽压(hPa)	平均相对湿度(%RH)	日降水	风速(最大)	平均风速(10m)	日照时数
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data = df[numeric_cols[0]].values
                else:
                    continue

            all_data_sequences.append(data)

        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {e}")

    # 检查数据长度是否足够
    total_length = sum(len(seq) for seq in all_data_sequences)
    print(f"总数据长度: {total_length}")

    if total_length < 625:  # 25x25=625
        print("警告: 数据长度不足以生成25x25的图像")
        return None

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


def create_sliding_windows_fixed(data_sequence, window_size=25, stride=25):
    """使用固定间隔的滑动窗口生成图像"""
    print("正在生成固定间隔滑动窗口图像...")

    windows = []
    total_length = len(data_sequence)

    # 计算需要的窗口数量
    num_windows = (total_length - window_size * window_size) // (stride * window_size) + 1
    print(f"预计生成窗口数量: {num_windows}")
    print(f"每个窗口大小: {window_size}x{window_size} = {window_size * window_size}个数据点")

    for i in tqdm(range(num_windows), desc="生成窗口"):
        start_idx = i * stride * window_size
        end_idx = start_idx + window_size * window_size

        if end_idx > total_length:
            break

        window_data = data_sequence[start_idx:end_idx]
        image = window_data.reshape(window_size, window_size)
        windows.append(image)

    if len(windows) == 0:
        print("警告: 没有生成任何窗口，数据可能太短")
        return np.array([])

    data_matrix = np.array(windows)
    print(f"实际生成的图像数量: {data_matrix.shape[0]}")
    print(f"图像形状: {data_matrix.shape}")

    return data_matrix


def create_datasets_with_nan_zero(data_matrix):
    """创建数据集，将有NaN的图像中的NaN值置零"""
    print("正在创建数据集并将NaN值置零...")

    if len(data_matrix) == 0:
        print("警告: 数据矩阵为空")
        return np.array([]), np.array([]), np.array([])

    has_nan = np.array([np.any(np.isnan(img)) for img in data_matrix])
    no_nan = ~has_nan

    print(f"包含NaN的图像数量: {np.sum(has_nan)}")
    print(f"不包含NaN的图像数量: {np.sum(no_nan)}")

    # 如果所有图像都包含NaN，我们需要处理这种情况
    if np.sum(no_nan) == 0:
        print("警告: 所有图像都包含NaN，将使用所有数据作为待补齐集")
        completion_data = data_matrix.copy()
        for i in range(len(completion_data)):
            nan_mask = np.isnan(completion_data[i])
            completion_data[i][nan_mask] = 0.0

        # 随机选择一些数据作为训练集和验证集
        if len(completion_data) > 10:
            train_indices = np.random.choice(len(completion_data), size=min(40, len(completion_data) // 2),
                                             replace=False)
            val_indices = np.random.choice([i for i in range(len(completion_data)) if i not in train_indices],
                                           size=min(20, len(completion_data) // 4), replace=False)

            train_data = completion_data[train_indices]
            val_data = completion_data[val_indices]
            completion_data = np.delete(completion_data, np.concatenate([train_indices, val_indices]), axis=0)
        else:
            train_data = completion_data[:len(completion_data) // 2]
            val_data = completion_data[len(completion_data) // 2:3 * len(completion_data) // 4]
            completion_data = completion_data[3 * len(completion_data) // 4:]
    else:
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


def apply_realistic_mask_optimized(data, num_station_groups=2, min_stations_per_group=1, max_stations_per_group=3,
                                   min_time_length=2, max_time_length=6, target_missing_ratio=0.02):
    """
    应用优化的真实掩膜模式，控制缺失比例在目标范围内
    """
    masked_data = data.copy()
    mask = np.ones_like(data)  # 1表示已知，0表示缺失

    time_length, num_stations = data.shape
    total_points = time_length * num_stations
    current_missing_ratio = 0

    max_attempts = 20  # 防止无限循环
    attempts = 0

    while current_missing_ratio < target_missing_ratio and attempts < max_attempts:
        attempts += 1

        # 随机选择一组站点（1-3个站点）
        stations_in_group = np.random.randint(min_stations_per_group, min(max_stations_per_group + 1, num_stations))
        station_indices = np.random.choice(num_stations, size=stations_in_group, replace=False)

        # 随机选择连续的时间段（2-6个时间步）
        time_length_group = np.random.randint(min_time_length, min(max_time_length + 1, time_length))
        time_start = np.random.randint(0, time_length - time_length_group)
        time_end = time_start + time_length_group

        # 计算应用此掩膜后的缺失比例
        new_missing_points = time_length_group * stations_in_group
        potential_new_ratio = (np.sum(1 - mask) + new_missing_points) / total_points

        # 如果添加后缺失比例不超过目标值的1.5倍，则应用掩膜
        if potential_new_ratio <= target_missing_ratio * 1.5:
            masked_data[time_start:time_end, station_indices] = 0
            mask[time_start:time_end, station_indices] = 0
            current_missing_ratio = np.sum(1 - mask) / total_points

    # 如果缺失比例仍然过低，添加一些小范围的缺失
    if current_missing_ratio < target_missing_ratio * 0.5:
        additional_attempts = 0
        while current_missing_ratio < target_missing_ratio and additional_attempts < 10:
            additional_attempts += 1
            # 单站点单时间点的缺失
            random_time = np.random.randint(0, time_length)
            random_station = np.random.randint(0, num_stations)
            if mask[random_time, random_station] == 1:  # 确保不是已经缺失的点
                masked_data[random_time, random_station] = 0
                mask[random_time, random_station] = 0
                current_missing_ratio = np.sum(1 - mask) / total_points

    print(f"实际缺失比例: {current_missing_ratio:.4f} (目标: {target_missing_ratio:.4f})")
    return masked_data, mask


def apply_masks_to_datasets_optimized(train_data, val_data, target_missing_ratio=0.02):
    """
    对训练集和验证集应用优化的掩膜处理，控制缺失比例
    """
    print("正在应用优化掩膜模式...")
    print(f"目标缺失比例: {target_missing_ratio:.2%}")

    masked_train_data = []
    train_masks = []
    masked_val_data = []
    val_masks = []

    # 对训练集应用掩膜
    if len(train_data) > 0:
        for i in tqdm(range(len(train_data)), desc="掩膜训练集"):
            original_img = train_data[i]
            masked_img, mask = apply_realistic_mask_optimized(
                original_img,
                num_station_groups=2,  # 减少组数
                min_stations_per_group=1,
                max_stations_per_group=2,  # 减少最大站点数
                min_time_length=1,
                max_time_length=4,  # 减少最大时间长度
                target_missing_ratio=target_missing_ratio
            )
            masked_train_data.append(masked_img)
            train_masks.append(mask)
    else:
        print("训练集为空，跳过掩膜处理")

    # 对验证集应用掩膜
    if len(val_data) > 0:
        for i in tqdm(range(len(val_data)), desc="掩膜验证集"):
            original_img = val_data[i]
            masked_img, mask = apply_realistic_mask_optimized(
                original_img,
                num_station_groups=2,
                min_stations_per_group=1,
                max_stations_per_group=2,
                min_time_length=1,
                max_time_length=4,
                target_missing_ratio=target_missing_ratio
            )
            masked_val_data.append(masked_img)
            val_masks.append(mask)
    else:
        print("验证集为空，跳过掩膜处理")

    if len(masked_train_data) > 0:
        masked_train_data = np.array(masked_train_data)
        train_masks = np.array(train_masks)
        actual_train_ratio = np.mean(1 - train_masks)
        print(f"训练集掩膜比例: {actual_train_ratio:.4f} (目标: {target_missing_ratio:.4f})")
    else:
        masked_train_data = np.array([])
        train_masks = np.array([])

    if len(masked_val_data) > 0:
        masked_val_data = np.array(masked_val_data)
        val_masks = np.array(val_masks)
        actual_val_ratio = np.mean(1 - val_masks)
        print(f"验证集掩膜比例: {actual_val_ratio:.4f} (目标: {target_missing_ratio:.4f})")
    else:
        masked_val_data = np.array([])
        val_masks = np.array([])

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
                              completion_dataset):
    """
    可视化真实掩膜模式
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('气象数据可视化 - 25x25优化掩膜模式 (控制缺失比例~2%)', fontsize=16, fontweight='bold')

    # 训练集可视化
    if len(train_data) >= 1:
        # 原始样本
        img_original = train_data[0]
        im0 = axes[0, 0].imshow(img_original, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'训练集原始样本\n形状: {img_original.shape}')
        axes[0, 0].set_xlabel('站点维度 (25个站点)')
        axes[0, 0].set_ylabel('时间维度 (25个时间步)')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 掩膜样本
        if len(masked_train_data) > 0:
            img_masked = masked_train_data[0]
            mask = train_masks[0]

            display_img = img_original.copy()
            mask_regions = mask == 0
            display_img[mask_regions] = np.nan

            im1 = axes[0, 1].imshow(display_img, cmap='viridis', aspect='auto')
            axes[0, 1].imshow(np.where(mask_regions, 1, np.nan), cmap='Reds', alpha=0.6, aspect='auto')

            masked_ratio = 1 - np.mean(mask)
            axes[0, 1].set_title(f'训练集掩膜样本\n掩膜比例: {masked_ratio:.2%}\n红色区域: 模拟缺失')
            axes[0, 1].set_xlabel('站点维度 (25个站点)')
            axes[0, 1].set_ylabel('时间维度 (25个时间步)')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        else:
            axes[0, 1].axis('off')
            axes[0, 1].text(0.5, 0.5, '训练集掩膜数据为空', ha='center', va='center', transform=axes[0, 1].transAxes)

    # 验证集可视化
    if len(val_data) >= 1:
        img_original = val_data[0]
        im0 = axes[1, 0].imshow(img_original, cmap='plasma', aspect='auto')
        axes[1, 0].set_title(f'验证集原始样本\n形状: {img_original.shape}')
        axes[1, 0].set_xlabel('站点维度 (25个站点)')
        axes[1, 0].set_ylabel('时间维度 (25个时间步)')
        plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

        if len(masked_val_data) > 0:
            img_masked = masked_val_data[0]
            mask = val_masks[0]

            display_img = img_original.copy()
            mask_regions = mask == 0
            display_img[mask_regions] = np.nan

            im1 = axes[1, 1].imshow(display_img, cmap='plasma', aspect='auto')
            axes[1, 1].imshow(np.where(mask_regions, 1, np.nan), cmap='Reds', alpha=0.6, aspect='auto')

            masked_ratio = 1 - np.mean(mask)
            axes[1, 1].set_title(f'验证集掩膜样本\n掩膜比例: {masked_ratio:.2%}\n红色区域: 模拟缺失')
            axes[1, 1].set_xlabel('站点维度 (25个站点)')
            axes[1, 1].set_ylabel('时间维度 (25个时间步)')
            plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        else:
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, '验证集掩膜数据为空', ha='center', va='center', transform=axes[1, 1].transAxes)

    # 待补齐集可视化
    if len(completion_dataset) >= 1:
        img = completion_dataset[0]
        # 检查哪些位置被置零（原始是NaN）
        zero_mask = (img == 0)

        display_img = img.copy()
        display_img[zero_mask] = np.nan

        im2 = axes[2, 0].imshow(display_img, cmap='coolwarm', aspect='auto')
        axes[2, 0].imshow(np.where(zero_mask, 1, np.nan), cmap='Blues', alpha=0.6, aspect='auto')

        zero_ratio = np.mean(zero_mask)
        axes[2, 0].set_title(f'待补齐集样本\n原始NaN比例: {zero_ratio:.2%}\n蓝色区域: 原始缺失')
        axes[2, 0].set_xlabel('站点维度 (25个站点)')
        axes[2, 0].set_ylabel('时间维度 (25个时间步)')
        plt.colorbar(im2, ax=axes[2, 0], fraction=0.046, pad=0.04)

    # 显示统计信息
    axes[2, 1].axis('off')
    stats_text = f"数据集统计 (25x25):\n"
    stats_text += f"训练集: {train_data.shape}\n"
    stats_text += f"验证集: {val_data.shape}\n"
    stats_text += f"待补齐集: {completion_dataset.shape}\n"
    if len(train_masks) > 0:
        train_missing_ratio = np.mean(1 - train_masks)
        stats_text += f"训练集掩膜比例: {train_missing_ratio:.2%}\n"
    if len(val_masks) > 0:
        val_missing_ratio = np.mean(1 - val_masks)
        stats_text += f"验证集掩膜比例: {val_missing_ratio:.2%}"
    axes[2, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                    transform=axes[2, 1].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()
    plt.show()


def save_masked_datasets(train_data, masked_train_data, train_masks,
                         val_data, masked_val_data, val_masks,
                         completion_dataset, scaler, output_folder='./weather_masked_datasets_25x25_optimized'):
    """保存掩膜后的数据集"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存原始数据
    if len(train_data) > 0:
        np.save(os.path.join(output_folder, 'train_original.npy'), train_data)
    if len(val_data) > 0:
        np.save(os.path.join(output_folder, 'val_original.npy'), val_data)

    # 保存掩膜数据
    if len(masked_train_data) > 0:
        np.save(os.path.join(output_folder, 'train_masked.npy'), masked_train_data)
        np.save(os.path.join(output_folder, 'train_masks.npy'), train_masks)
    if len(masked_val_data) > 0:
        np.save(os.path.join(output_folder, 'val_masked.npy'), masked_val_data)
        np.save(os.path.join(output_folder, 'val_masks.npy'), val_masks)

    # 保存待补齐集
    if len(completion_dataset) > 0:
        np.save(os.path.join(output_folder, 'completion_dataset.npy'), completion_dataset)

    # 保存归一化参数
    np.save(os.path.join(output_folder, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(output_folder, 'scaler_scale.npy'), scaler.scale_)

    stats = {
        'train_original_shape': train_data.shape if len(train_data) > 0 else 'Empty',
        'train_masked_shape': masked_train_data.shape if len(masked_train_data) > 0 else 'Empty',
        'val_original_shape': val_data.shape if len(val_data) > 0 else 'Empty',
        'val_masked_shape': masked_val_data.shape if len(masked_val_data) > 0 else 'Empty',
        'completion_shape': completion_dataset.shape if len(completion_dataset) > 0 else 'Empty',
        'train_masked_ratio': np.mean(1 - train_masks) if len(train_masks) > 0 else 0,
        'val_masked_ratio': np.mean(1 - val_masks) if len(val_masks) > 0 else 0,
        'completion_zero_ratio': np.mean(completion_dataset == 0) if len(completion_dataset) > 0 else 0,
        'normalization_mean': scaler.mean_[0],
        'normalization_std': scaler.scale_[0]
    }

    with open(os.path.join(output_folder, 'dataset_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    print(f"掩膜数据集已保存到: {output_folder}")


def generate_synthetic_long_sequence_optimized():
    """生成优化的合成数据，减少NaN比例"""
    print("生成优化的合成长序列气象数据...")

    np.random.seed(42)
    total_length = 100000
    time_axis = np.linspace(0, 20 * np.pi, total_length)

    base_sequence = (np.sin(time_axis) +
                     0.5 * np.sin(2 * time_axis) +
                     0.2 * np.sin(0.5 * time_axis) +
                     0.1 * time_axis / (2 * np.pi) +
                     0.3 * np.random.randn(total_length))

    # 大幅减少NaN的比例
    data_with_nan = base_sequence.copy()
    nan_probability = 0.005  # 从0.01降低到0.005

    nan_positions = np.random.choice(total_length, size=int(total_length * nan_probability), replace=False)
    data_with_nan[nan_positions] = np.nan

    # 减少连续缺失块
    for _ in range(5):  # 从10减少到5
        block_start = np.random.randint(0, total_length - 15)
        block_length = np.random.randint(2, 8)  # 减少块长度
        data_with_nan[block_start:block_start + block_length] = np.nan

    actual_nan_ratio = np.sum(np.isnan(data_with_nan)) / len(data_with_nan)
    print(f"合成数据NaN比例: {actual_nan_ratio:.2%}")
    return data_with_nan


def main():
    csv_folder_path = "I:/datasets/1"#I:/datasets/1
    #C:/Users/张宸嘉/PycharmProjects/Informer2020-main/PET预测项目/model/weather_stations_output

    try:
        data_sequence = load_all_csv_data(csv_folder_path)
        if data_sequence is None:
            print("使用合成数据...")
            data_sequence = generate_synthetic_long_sequence_optimized()
    except Exception as e:
        print(f"加载真实数据失败: {e}")
        print("使用合成数据...")
        data_sequence = generate_synthetic_long_sequence_optimized()

    normalized_data, scaler = normalize_data(data_sequence)
    data_matrix = create_sliding_windows_fixed(normalized_data, window_size=25, stride=25)

    if len(data_matrix) == 0:
        print("无法生成数据矩阵，程序退出")
        return

    train_data, val_data, completion_dataset = create_datasets_with_nan_zero(data_matrix)

    # 应用优化掩膜，目标缺失比例2%
    masked_train_data, train_masks, masked_val_data, val_masks = apply_masks_to_datasets_optimized(
        train_data, val_data, target_missing_ratio=0.02
    )

    # 分析掩膜模式
    if len(train_masks) > 0:
        print("\n=== 训练集掩膜模式分析 ===")
        analyze_mask_pattern(train_masks[0])

    visualize_realistic_masks(
        train_data, masked_train_data, train_masks,
        val_data, masked_val_data, val_masks,
        completion_dataset
    )

    save_masked_datasets(
        train_data, masked_train_data, train_masks,
        val_data, masked_val_data, val_masks,
        completion_dataset, scaler
    )

    print("\n=== 数据集统计 ===")
    total_samples = (len(train_data) if len(train_data) > 0 else 0) + \
                    (len(val_data) if len(val_data) > 0 else 0) + \
                    (len(completion_dataset) if len(completion_dataset) > 0 else 0)
    print(f"总样本数: {total_samples}")

    # 打印详细的缺失统计
    if len(train_masks) > 0:
        train_missing_ratio = np.mean(1 - train_masks)
        print(f"训练集实际缺失比例: {train_missing_ratio:.4f}")
    if len(val_masks) > 0:
        val_missing_ratio = np.mean(1 - val_masks)
        print(f"验证集实际缺失比例: {val_missing_ratio:.4f}")


if __name__ == "__main__":
    main()