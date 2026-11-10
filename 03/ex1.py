import torch
from torch import nn

if __name__ == "__main__":
#p1
#Tensorを定義
#(batchsize,channel,width,height)
    my_tensor = torch.ones((32, 3, 128, 128))
    print(f"original:{my_tensor.shape}")

    #p2
    #畳み込み定義と適用
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
    out1 = conv(my_tensor)
    print(f"out1:{out1.shape}")

    #p3
    conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=1)
    out2 = conv(my_tensor)
    print(f"out2:{out2.shape}")

    #p4
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
    out3 = conv(my_tensor)
    print(f"out3:{out3.shape}")
    conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=2, padding=2)
    out4 = conv(my_tensor)
    print(f"out4:{out4.shape}")
