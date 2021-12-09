import torch.nn as nn
import torchvision.models as models


class NetGen(nn.Module):
    '''Generator'''
    def __init__(self):
        super(NetGen, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bnorm5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.1)

        self.deconv6 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bnorm6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bnorm7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

        self.deconv8 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bnorm8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()

        self.deconv9 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bnorm9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()

        self.deconv10 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.tanh = nn.Tanh()
        

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bnorm1(h)
        h = self.relu1(h) 
        pool1 = h

        h = self.conv2(h)
        h = self.bnorm2(h)
        h = self.relu2(h) 
        pool2 = h

        h = self.conv3(h) 
        h = self.bnorm3(h)
        h = self.relu3(h)
        pool3 = h

        h = self.conv4(h) 
        h = self.bnorm4(h)
        h = self.relu4(h)
        pool4 = h

        h = self.conv5(h) 
        h = self.bnorm5(h)
        h = self.relu5(h)

        h = self.deconv6(h)
        h = self.bnorm6(h)
        h = self.relu6(h) 
        h = h + pool4

        h = self.deconv7(h)
        h = self.bnorm7(h)
        h = self.relu7(h) 
        h = h + pool3

        h = self.deconv8(h)
        h = self.bnorm8(h)
        h = self.relu8(h)
        h = h + pool2

        h = self.deconv9(h)
        h = self.bnorm9(h)
        h = self.relu9(h)
        h = h + pool1

        h = self.deconv10(h)
        h = self.tanh(h) 
        return h

class NetDis(nn.Module):
    '''Discriminator'''
    def __init__(self):
        super(NetDis, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),


            nn.Conv2d(512, 512, 8, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)