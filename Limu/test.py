import torch
from torch.utils.data import DataLoader, TensorDataset

def main():
    # 创建一个简单的数据集
    data = torch.randn(100, 10)  # 100个样本，每个样本10个特征
    targets = torch.randint(0, 2, (100,))  # 100个标签，二分类

    # 将数据和标签组合成一个Dataset
    dataset = TensorDataset(data, targets)

    # 创建一个DataLoader
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # 使用DataLoader加载数据
    for batch_data, batch_targets in data_loader:
        # 处理每个批次的数据
        print(batch_data.shape, batch_targets.shape)
    
if __name__ == '__main__':
    main()