import os
import pandas as pd
import torch

print("******2.2.1. 读取数据集******")
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

print("******2.2.2. 处理缺失值******")
inputs_num, inputs_str, outputs = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
inputs_num = inputs_num.fillna(inputs_num.mean())
inputs = pd.concat((inputs_num, inputs_str), axis=1)
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

print("******2.2.3. 转换为张量格式******")
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y)