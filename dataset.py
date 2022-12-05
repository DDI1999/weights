from torch.utils.data import Dataset
import os
from PIL import Image


# root_dir: ./data/train  ./data/test
class MyDateset(Dataset):  # 继承Dataset
    def __init__(self, root, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root  # 文件目录
        self.class_index = {cla: ind for ind, cla in enumerate(os.listdir(self.root_dir))}  # {class : index}

        lists = []
        for i in os.listdir(self.root_dir):
            image_ = os.listdir(os.path.join(self.root_dir, i))  # train/ll
            for img in image_:
                lists.append(os.path.join(self.root_dir, i, img))
        self.images = lists  # root_dir目录里的所有图片['jpg']
        self.transform = transform  # 变换

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    # [(img,index)]
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        img = Image.open(self.images[index])  # 根据索引index获取该图片
        label = self.class_index[self.images[index].split('\\')[-2]]

        if self.transform is not None:  # transform
            img = self.transform(img)
        return img, label  # 返回该样本
