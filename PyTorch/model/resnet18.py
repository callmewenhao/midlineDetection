import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim,
                               out_channels=out_dim, 
                               kernel_size=3, 
                               stride=stride, 
                               padding=1)
                               
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim,
                               out_channels=out_dim, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        if stride == 2 or in_dim != out_dim:
            self.downsample = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1, stride=stride),
                                            nn.BatchNorm2d(out_dim))
        else:
            self.downsample = Identity()


    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(h)
        x = x + identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, in_dim=64, num_classes=5):
        super().__init__()

        self.in_dim = in_dim

        # stem layers
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=in_dim, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU()


        # blocks layers
        self.layers1 = self._make_layer(dim=64, n_blocks=2, stride=1)
        self.layers2 = self._make_layer(dim=128, n_blocks=2, stride=2)
        self.layers3 = self._make_layer(dim=256, n_blocks=2, stride=2)
        self.layers4 = self._make_layer(dim=512, n_blocks=2, stride=2)


        # head layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)



    def _make_layer(self, dim, n_blocks, stride):
        """ 构建 block """
        layer_list = []
        layer_list.append(Block(self.in_dim, dim, stride=stride))
        self.in_dim = dim
        for i in range(1, n_blocks):
            layer_list.append(Block(self.in_dim, dim, stride=1))
        return nn.Sequential(*layer_list)  # *会将列表拆分成一个一个的独立元素


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.avgpool(x)
        # print(x.shape)  # [512, 1, 1]
        # x = x.flatten(1)
        x = x.reshape(-1, 2, 256)
        # print(x.shape)
        x = self.classifier(x)
        return x


def main():
    # print("Hello, world!\n")
    t = torch.randn([8, 1, 60, 94])
    model = ResNet18()
    # print(model)
    out = model(t)
    print(out.shape)

if __name__ == "__main__":
    main()
