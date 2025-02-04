import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# =========================
# 1. 数据加载
# =========================
# 从文件中加载数据，数据已经为 torch.Tensor 类型
data = torch.load('record_data.pth')
obj_pos  = data["obj_pos"]   # shape: (N, 3)
obj_ori  = data["obj_ori"]   # shape: (N, 4)
succ_idx = data["succ_idx"]  # shape: (N, 1)，作为二分类标签，值通常为 0 或 1

# print(len(obj_pos), len(obj_ori), len(succ_idx))

# print(obj_pos[:10], obj_ori[:10], succ_idx[:10])

# exit()

# 将succ_idx为1的数据拿出来
index = succ_idx == 1
# print("succ_idx为1的数据量：", index.sum().item())
# exit()
obj_pos_succ = obj_pos[index]
obj_ori_succ = obj_ori[index]
# 从succ_idx为0的数据中随机选取与succ_idx为1的数据量相同的数据
index = succ_idx == 0
obj_pos_fail = obj_pos[index]
obj_ori_fail = obj_ori[index]
obj_pos = torch.cat([obj_pos_succ, obj_pos_fail[:len(obj_pos_succ)]], dim=0)
obj_ori = torch.cat([obj_ori_succ, obj_ori_fail[:len(obj_ori_succ)]], dim=0)
succ_idx = torch.cat([torch.ones(len(obj_pos_succ)), torch.zeros(len(obj_pos_succ))], dim=0)
print(len(obj_pos), len(obj_ori), len(succ_idx))
exit()
# =========================
# 2. 自定义数据集
# =========================
class MyDataset(Dataset):
    def __init__(self, obj_pos, obj_ori, succ_idx):
        # 拼接两个输入，形成 (N, 7) 的输入特征
        self.x = torch.cat([obj_pos, obj_ori], dim=1)
        # 将标签转换为 float 类型（BCEWithLogitsLoss 要求标签为 float 类型）
        self.y = succ_idx.float()
        
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 实例化数据集
dataset = MyDataset(obj_pos, obj_ori, succ_idx)

# =========================
# 3. 数据集划分（8:2比例）
# =========================
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

# 使用 random_split 随机划分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader，其中训练集打乱数据，测试集不需要打乱
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1024, shuffle=True)

# =========================
# 4. 定义模型
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
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)    # 输出 1 个数，后续使用 Sigmoid 进行二分类判断
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNet()

# 如果有 GPU，将模型转移到对应设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# 5. 定义损失函数与优化器
# =========================
# 使用 BCEWithLogitsLoss，内部会先对输出应用 Sigmoid 再计算二元交叉熵
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 6. 训练过程（每个 epoch 后在测试集上评估）
# =========================
num_epochs = 100

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    epoch_loss = 0.0
    total_correct = 0
    total_samples_train = 0

    # 训练过程
    for batch_inputs, batch_labels in train_loader:
        # 将数据转移到设备（例如 GPU）
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()                   # 清空梯度
        outputs = model(batch_inputs)           # 前向传播，输出 shape 为 (batch_size, 1)
        # 计算损失时确保标签 shape 为 (batch_size, 1)
        loss = criterion(outputs, batch_labels.unsqueeze(1))
        loss.backward()                         # 反向传播
        optimizer.step()                        # 更新参数
        
        epoch_loss += loss.item() * batch_inputs.size(0)
        
        # 计算本 batch 的准确率
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        total_correct += (preds == batch_labels.unsqueeze(1)).sum().item()
        total_samples_train += batch_labels.size(0)
    
    epoch_loss /= train_size
    train_accuracy = total_correct / total_samples_train

    # 测试过程
    model.eval()
    test_loss = 0.0
    test_correct = 0
    total_samples_test = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels.unsqueeze(1))
            test_loss += loss.item() * batch_inputs.size(0)
            
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            test_correct += (preds == batch_labels.unsqueeze(1)).sum().item()
            total_samples_test += batch_labels.size(0)
    
    test_loss /= test_size
    test_accuracy = test_correct / total_samples_test

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

# =========================
# 7. 模型保存
# =========================
torch.save(model.state_dict(), "simple_net_classification.pth")
print("训练完成，模型已保存。")
