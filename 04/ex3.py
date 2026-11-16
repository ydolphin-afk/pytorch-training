from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, dataset_dir):
        dir_path = Path(dataset_dir).resolve()
        self.img_list = list(dir_path.glob("**/*.png"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        img_path = Path(img_path)
        parts = img_path.parts
        label = int(parts[-2])

        return img_tensor, label

if __name__ == "__main__":
    my_dataset = MyDataset("04/data")
    img, label = my_dataset[15]
    #1.サイズ
    print(img.size())
    #2.ラベル
    print(label)