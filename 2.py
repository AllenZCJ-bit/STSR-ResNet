import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')


# 数据加载函数
def load_datasets(data_folder='./weather_masked_datasets_25x25_optimized'):#weather_masked_datasets_25x25_optimized,,,,weather_masked_datasets
    """加载数据集"""
    print("正在加载数据集...")

    # 如果默认路径不存在，尝试其他路径
    if not os.path.exists(data_folder):
        possible_paths = [
            './weather_masked_datasets',
            'weather_masked_datasets',
            '../weather_masked_datasets',
            '../../weather_masked_datasets'
        ]
        for path in possible_paths:
            if os.path.exists(os.path.join(path, 'train_original.npy')):
                data_folder = path
                print(f"找到数据文件夹: {data_folder}")
                break
        else:
            raise FileNotFoundError("未找到数据文件，请确保已运行 datagen.py")

    print(f"使用数据文件夹: {data_folder}")

    # 加载所有文件
    train_original = torch.tensor(np.load(os.path.join(data_folder, 'train_original.npy')), dtype=torch.float32)
    train_masked = torch.tensor(np.load(os.path.join(data_folder, 'train_masked.npy')), dtype=torch.float32)
    train_masks = torch.tensor(np.load(os.path.join(data_folder, 'train_masks.npy')), dtype=torch.float32)

    val_original = torch.tensor(np.load(os.path.join(data_folder, 'val_original.npy')), dtype=torch.float32)
    val_masked = torch.tensor(np.load(os.path.join(data_folder, 'val_masked.npy')), dtype=torch.float32)
    val_masks = torch.tensor(np.load(os.path.join(data_folder, 'val_masks.npy')), dtype=torch.float32)

    # 添加通道维度
    train_original = train_original.unsqueeze(1)
    train_masked = train_masked.unsqueeze(1)
    train_masks = train_masks.unsqueeze(1)

    val_original = val_original.unsqueeze(1)
    val_masked = val_masked.unsqueeze(1)
    val_masks = val_masks.unsqueeze(1)

    print(f"训练集: {train_original.shape}")
    print(f"验证集: {val_original.shape}")

    train_data = (train_original, train_masked, train_masks)
    val_data = (val_original, val_masked, val_masks)

    return train_data, val_data

#Resnet
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()

        # 第一层卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        # 第一差分项
        self.diff1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1)
        )

        # 第二层卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # 第二差分项
        self.diff2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1)
        )

        # 特征融合和输出层
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),  # 1x1卷积降维
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)  # 输出单通道
        )

        # 可选的线性层（在空间维度上应用）
        self.spatial_fc = nn.Sequential(
            nn.Conv2d(128, 64, 1),  # 替代全连接层的1x1卷积
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)  # 最终输出
        )

    def forward(self, x):
        # 第一卷积块
        x1 = self.conv_block1(x)

        # 第一差分项 + 残差
        diff1 = self.diff1(x1)
        x1_res = x1 + diff1  # 残差连接

        # 第二卷积块
        x2 = self.conv_block2(x1_res)

        # 第二差分项 + 残差
        diff2 = self.diff2(x2)
        x2_res = x2 + diff2  # 残差连接

        # 选择特征融合方式
        # 方式1: 传统卷积融合
        output = self.feature_fusion(x2_res)

        # 方式2: 使用1x1卷积模拟全连接（保持空间信息）
        # output = self.spatial_fc(x2_res)

        return output
# #简单卷积网络模型
# class SimpleConvNet(nn.Module):
#     def __init__(self):
#         super(SimpleConvNet, self).__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, 3, padding=1)
#         )
#
#     def forward(self, x):
#         return self.network(x)


# #时空残差卷积网络
# class SimpleConvNet(nn.Module):
#     def __init__(self, input_size=105):
#         super(SimpleConvNet, self).__init__()
#         self.input_size = input_size
#
#         # 第一层卷积块
#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU()
#         )
#         self.conv_2d = nn.Sequential(
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.ReLU()
#         )
#         # 第一组时空残差 - 处理原始输入x (1通道)
#         self.spatial_residual1 = nn.Sequential(
#             nn.Conv1d(1, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(64, 64, 3, padding=1)  # 输出64通道匹配x1
#         )
#
#         self.temporal_residual1 = nn.Sequential(
#             nn.Conv1d(1, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(64, 64, 3, padding=1)  # 输出64通道匹配x1
#         )
#
#         # 第二层卷积块
#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.ReLU()
#         )
#
#         # 第二组时空残差 - 处理x1_with_residual (64通道)
#         self.spatial_residual2 = nn.Sequential(
#             nn.Conv1d(64, 128, 3, padding=1),  # 输入64，输出128匹配x2
#             nn.ReLU(),
#             nn.Conv1d(128, 128, 3, padding=1)  # 输出128通道匹配x2
#         )
#
#         self.temporal_residual2 = nn.Sequential(
#             nn.Conv1d(64, 128, 3, padding=1),  # 输入64，输出128匹配x2
#             nn.ReLU(),
#             nn.Conv1d(128, 128, 3, padding=1)  # 输出128通道匹配x2
#         )
#
#         # 特征融合和输出层
#         self.feature_fusion = nn.Sequential(
#             nn.Conv2d(128, 64, 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 1, 3, padding=1)
#         )

    # def compute_spatiotemporal_residual(self, x, spatial_res_net, temporal_res_net, output_channels):
    #     """
    #     计算时空残差
    #     output_channels: 输出的通道数，用于匹配主干网络
    #     """
    #     batch_size, channels, height, width = x.shape
    #
    #     # 空间特征提取
    #     spatial_features = x.mean(dim=2)  # [batch, channels, width]
    #     spatial_vector = spatial_res_net(spatial_features)  # [batch, output_channels, width]
    #
    #     # 时间特征提取
    #     temporal_features = x.mean(dim=3)  # [batch, channels, height]
    #     temporal_vector = temporal_res_net(temporal_features)  # [batch, output_channels, height]
    #
    #     # 外积得到残差矩阵
    #     residual_matrix = torch.zeros(batch_size, output_channels, height, width, device=x.device)
    #
    #     for b in range(batch_size):
    #         for c in range(output_channels):
    #             spatial_vec = spatial_vector[b, c]  # [width]
    #             temporal_vec = temporal_vector[b, c]  # [height]
    #
    #             # 外积
    #             matrix = torch.outer(temporal_vec, spatial_vec)  # [height, width]
    #             residual_matrix[b, c] = matrix
    #
    #     return residual_matrix
    #
    # def forward(self, x):
    #     # 第一卷积块
    #     x1 = self.conv_block1(x)
    #
    #     # 计算第一时空残差：基于原始输入x，输出64通道匹配x1
    #     residual1 = self.conv_2d(x)
    #     x1_with_residual = x1 + residual1
    #
    #     # 第二卷积块
    #     x2 = self.conv_block2(x1_with_residual)
    #
    #     # 计算第二时空残差：基于x1_with_residual，输出128通道匹配x2
    #     residual2 = self.compute_spatiotemporal_residual(x1_with_residual, self.spatial_residual2,
    #                                                      self.temporal_residual2, 128)
    #     x2_with_residual = x2 + residual2
    #
    #     # 特征融合
    #     output = self.feature_fusion(x2_with_residual)
    #
    #     return output
# 训练函数MSE
def train_simple_conv():
    """训练简单的卷积网络"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_data, val_data = load_datasets()
    train_original, train_masked, train_masks = train_data
    val_original, val_masked, val_masks = val_data

    # 移动到设备
    train_original = train_original.to(device)
    train_masked = train_masked.to(device)
    train_masks = train_masks.to(device)
    val_original = val_original.to(device)
    val_masked = val_masked.to(device)
    val_masks = val_masks.to(device)

    # 创建模型
    model = SimpleConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    print("开始训练简单卷积网络...")

    for epoch in range(50):  # 增加训练轮数
        model.train()
        total_loss = 0

        # 使用小批量训练
        batch_size = 8
        num_batches = (len(train_masked) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_masked))

            inputs = train_masked[start_idx:end_idx]
            targets = train_original[start_idx:end_idx]
            masks = train_masks[start_idx:end_idx]

            outputs = model(inputs)
            # 只在掩膜区域计算损失
            loss = criterion(outputs * (1 - masks), targets * (1 - masks))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_masked), batch_size):
                inputs = val_masked[i:i + batch_size]
                targets = val_original[i:i + batch_size]
                masks = val_masks[i:i + batch_size]

                outputs = model(inputs)
                loss = criterion(outputs * (1 - masks), targets * (1 - masks))
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/40], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    print("训练完成!")
    return model, train_losses, val_losses


# 可视化函数
def visualize_comparison(original, masked, predicted, mask, model_name="模型", sample_idx=0):
    """可视化比较结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - 数据补全结果', fontsize=16, fontweight='bold')

    # 转换为numpy
    original_np = original[sample_idx, 0].cpu().numpy()
    masked_np = masked[sample_idx, 0].cpu().numpy()
    predicted_np = predicted[sample_idx, 0].cpu().numpy()
    mask_np = mask[sample_idx, 0].cpu().numpy()

    # 原始图像
    im0 = axes[0, 0].imshow(original_np, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].set_xlabel('空间维度')
    axes[0, 0].set_ylabel('时间维度')
    plt.colorbar(im0, ax=axes[0, 0])

    # 掩膜图像
    display_masked = original_np.copy()
    mask_regions = mask_np == 0
    display_masked[mask_regions] = np.nan

    im1 = axes[0, 1].imshow(display_masked, cmap='viridis', aspect='auto')
    axes[0, 1].imshow(np.where(mask_regions, 1, np.nan), cmap='Reds', alpha=0.6, aspect='auto')
    axes[0, 1].set_title('输入掩膜图像\n红色区域: 缺失区域')
    axes[0, 1].set_xlabel('空间维度')
    axes[0, 1].set_ylabel('时间维度')
    plt.colorbar(im1, ax=axes[0, 1])

    # 预测图像
    im2 = axes[1, 0].imshow(predicted_np, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('预测图像')
    axes[1, 0].set_xlabel('空间维度')
    axes[1, 0].set_ylabel('时间维度')
    plt.colorbar(im2, ax=axes[1, 0])

    # 误差热力图
    error = np.abs(predicted_np - original_np)
    error_display = error.copy()
    error_display[mask_np == 1] = 0  # 已知区域误差设为0

    im3 = axes[1, 1].imshow(error_display, cmap='hot', aspect='auto', vmin=0)
    axes[1, 1].set_title('误差热力图\n(仅显示缺失区域误差)')
    axes[1, 1].set_xlabel('空间维度')
    axes[1, 1].set_ylabel('时间维度')
    plt.colorbar(im3, ax=axes[1, 1])

    # 计算并显示掩膜区域MSE
    if np.any(mask_regions):
        mask_mse = np.mean((predicted_np[mask_regions] - original_np[mask_regions]) ** 2)
        axes[1, 1].text(0.02, 0.98, f'掩膜区域MSE: {mask_mse:.6f}',
                        transform=axes[1, 1].transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    if np.any(mask_regions):
        mask_mae = np.mean((predicted_np[mask_regions] - original_np[mask_regions]) )
        axes[1, 1].text(0.02, 0.98, f'掩膜区域MAE: {mask_mae:.6f}',
                        transform=axes[1, 1].transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.show()

    return mask_mse if np.any(mask_regions) else 0,mask_mae if np.any(mask_regions) else 0


# 测试函数
def test_model(model, val_data, scaler, num_samples=3):
    """测试模型并可视化结果（包含反归一化）"""
    device = next(model.parameters()).device
    val_original, val_masked, val_masks = val_data

    # 移动到设备
    val_original = val_original.to(device)
    val_masked = val_masked.to(device)
    val_masks = val_masks.to(device)

    model.eval()
    all_mask_mse = []
    all_mask_mse_original = []  # 反归一化后的MSE

    print(f"\n测试 {num_samples} 个样本...")

    with torch.no_grad():
        for i in range(min(num_samples+1, len(val_original))):
            print(f"\n样本 {i + 1}:")

            input_data = val_masked[i:i + 1]
            target_data = val_original[i:i + 1]
            mask_data = val_masks[i:i + 1]

            # 预测
            predicted = model(input_data)

            # 反归一化
            target_original = scaler.inverse_transform(
                target_data.cpu().numpy().flatten().reshape(-1, 1)
            ).reshape(target_data.shape)

            predicted_original = scaler.inverse_transform(
                predicted.cpu().numpy().flatten().reshape(-1, 1)
            ).reshape(predicted.shape)

            # 计算归一化尺度上的MSE（用于训练监控）
            mask_mse_normalized = np.mean(
                (predicted.cpu().numpy()[mask_data.cpu().numpy() == 0] -
                 target_data.cpu().numpy()[mask_data.cpu().numpy() == 0]) ** 2
            )

            # 计算原始尺度上的MSE（用于论文报告）
            mask_mse_original = np.mean(
                (predicted_original[mask_data.cpu().numpy() == 0] -
                 target_original[mask_data.cpu().numpy() == 0]) ** 2
            )

            # 可视化结果（使用原始尺度数据）
            _ = visualize_comparison(
                torch.tensor(target_original),
                torch.tensor(scaler.inverse_transform(input_data.cpu().numpy().flatten().reshape(-1, 1)).reshape(
                    input_data.shape)),
                torch.tensor(predicted_original),
                mask_data,
                model_name="简单卷积网络", sample_idx=0
            )

            all_mask_mse.append(mask_mse_normalized)
            all_mask_mse_original.append(mask_mse_original)

            print(f"归一化尺度掩膜区域MSE: {mask_mse_normalized:.6f}")
            print(f"原始尺度掩膜区域MSE: {mask_mse_original:.6f}")
            print(f"原始尺度RMSE: {np.sqrt(mask_mse_original):.4f}°C")  # 假设是温度数据

    if all_mask_mse_original:
        avg_mse_original = np.mean(all_mask_mse_original)
        avg_rmse_original = np.sqrt(avg_mse_original)
        print(f"\n平均原始尺度掩膜区域MSE: {avg_mse_original:.6f}")
        print(f"平均原始尺度RMSE: {avg_rmse_original:.4f}°C")

    return all_mask_mse, all_mask_mse_original


# 主函数
def main():
    """主函数（修改版，包含scaler传递）"""
    print("=== 气象数据补全 - 简单卷积网络 ===")

    try:
        # 加载scaler
        data_folder = './weather_masked_datasets_25x25_optimized'#weather_masked_datasets_25x25_optimized,,,,weather_masked_datasets
        scaler_mean = np.load(os.path.join(data_folder, 'scaler_mean.npy'))
        scaler_scale = np.load(os.path.join(data_folder, 'scaler_scale.npy'))

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = scaler_mean
        scaler.scale_ = scaler_scale

        # 训练模型
        model, train_losses, val_losses= train_simple_conv()

        # 绘制训练曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('简单卷积网络 - 训练曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 加载验证数据
        _, val_data = load_datasets()

        # 测试模型（传递scaler）
        test_results_normalized, test_results_original = test_model(
            model, val_data, scaler, num_samples=3
        )

        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale
        }, 'simple_conv_model.pth')

        print("\n模型和归一化参数已保存为 'simple_conv_model.pth'")

        return model, train_losses, val_losses, test_results_original

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# 如果直接运行这个文件，执行主函数
if __name__ == "__main__":
    main()