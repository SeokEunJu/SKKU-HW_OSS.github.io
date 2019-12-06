import torch.nn as nn
import torch

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


def load_FAN():
    '''
    input: (256, 256, 3) image with scale [0, 1]
    output: (256, 256, 68) face landmarks
    '''
    from FAN.models import FAN
    import torch.utils.model_zoo as model_zoo
    print('========== Loading FAN model ==========')
    model = FAN(2)

    weights = model_zoo.load_url('https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar', map_location=lambda storage, loc: storage)

    # cutoff two hourglass network
    pretrained_dict = {k: v for k, v in weights.items() if k in model.state_dict()}

    model.load_state_dict(pretrained_dict)
    model.eval()

    return model


class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()

        self.upsample = torch.nn.UpsamplingBilinear2d([256, 256])

    def forward(self, input):
        return (self.upsample(input) + 1.) / 2
