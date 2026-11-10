import torch
from torch import nn

class MyModel(nn.Module):
    #目標の出力サイズを引数に
    def __init__(self, out_features=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=8, padding=0)
        #conv層から受け取る出力チャネルに応じて設定
        self.bn = nn.BatchNorm2d(num_features=256) 
        self.relu =nn.ReLU()
        #Linearは1次元しか受け取らないからデータを加工
        input_features = 256 * 16 * 16
        self.fc = nn.Linear(in_features=input_features, out_features=out_features)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    in_tensor = torch.ones(32, 3, 128, 128)
    model = MyModel()
    out = model(in_tensor)
    print(out.shape)
