import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


    def forward(self, x):

        return


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()


    def forward(self, x):

        return


# residual block - not required, but recommendable
# identical structure of this block is repeated in the Generator
class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()


    def forward(self, x):

        return


class example_generator(nn.Module):
    # initial setting of the network
    # name each part of the network
    def __init__(self):
        super(example_generator, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU()
        )


    # how an input should go through the network
    # ex: input x goes through layer_1, and layer_2. returns the value after that
    def forward(self, x):
        x = x.float()
        x = self.layer_1(x)
        x = self.layer_2(x)

        return x
