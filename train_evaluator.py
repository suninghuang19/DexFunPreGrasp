import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 数据加载
# =========================
# 数据存储在 'record_data.pth' 中，数据已经是 torch.Tensor 类型
data = torch.load('record_data.pth')
obj_pos = data["obj_pos"]  # shape: (N, 3)
obj_ori = data["obj_ori"]  # shape: (N, 4)
cost    = data["cost"]     # shape: (N, 1) 作为回归任务的目标

#将cost大于10的数据去掉
index = cost < 10
obj_pos = obj_pos[index]
obj_ori = obj_ori[index]
cost = cost[index]

# =========================
# 2. 自定义数据集
# =========================
class MyDataset(Dataset):
    def __init__(self, obj_pos, obj_ori, cost):
        # 直接拼接两个输入张量，结果形状为 (N, 7)
        self.x = torch.cat([obj_pos, obj_ori], dim=1)
        self.y = cost

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = MyDataset(obj_pos, obj_ori, cost)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

# =========================
# 3. 定义模型
# =========================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 1024),  # 输入维度为 7 (3+4)
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)    # 输出维度为 1
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNet()

# 如果有 GPU，可以使用如下方式将模型和数据转到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# 4. 定义损失函数与优化器
# =========================
criterion = nn.MSELoss()            # 均方误差损失，适用于回归任务
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 5. 训练过程
# =========================
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_inputs, batch_labels in dataloader:
        # 如果使用 GPU，则需要将 batch_inputs 与 batch_labels 转移到 device：
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()               # 清空梯度
        outputs = model(batch_inputs)       # 前向传播
        loss = criterion(outputs, batch_labels)  # 计算损失
        loss.backward()                     # 反向传播
        optimizer.step()                    # 参数更新

        epoch_loss += loss.item() * batch_inputs.size(0)
    
    epoch_loss /= len(dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# =========================
# 6. 模型保存
# =========================
torch.save(model.state_dict(), "simple_net.pth")
print("训练完成，模型已保存。")
