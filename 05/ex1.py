import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

if __name__ == "__main__":
    image_path = "05/ex_data/dog_img.png"
    image = Image.open(image_path)

    transform1 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        ])

    transform3 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        ])

    #テンソル変換
    transformed_image = transform1(image)    
    print("変換前画像サイズ")
    print(image.size)
    print("変換後画像サイズ")
    print(transformed_image.size())
    plt.imshow(transformed_image.permute(1,2,0))
    plt.show()
    transformed_image = transform2(image)    
    plt.imshow(transformed_image.permute(1,2,0))
    plt.show()
    transformed_image = transform3(image)    
    plt.imshow(transformed_image.permute(1,2,0))
    plt.show()
