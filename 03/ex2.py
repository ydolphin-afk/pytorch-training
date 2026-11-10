import torch
from torch import nn

if __name__ == "__main__":
    #入力Tensorの定義
    _in = torch.ones((32, 1024))
    print(f"_in : {_in.shape}")

    #全結合定義、適用
    fc = nn.Linear(in_features=1024, out_features=256, bias=True)
    out1 = fc(_in)
    print(f"out:{out1.shape}")
    fc = nn.Linear(in_features=1024, out_features=2048, bias=True)
    out2 = fc(_in)
    print(f"out:{out2.shape}")    
    out3 = out1.reshape(32, 16, 16)
    print(f"out3 : {out3.shape}")