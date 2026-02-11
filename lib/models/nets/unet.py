import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from lib.models.modules.be_bc_block import BELModule, BEBModule, BCModule


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))
    

class OutConv_nosig(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_nosig, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self,configer):
        super(UNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, self.num_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)

        return output
    


class UNet_BE(nn.Module):
    def __init__(self, configer):
        super(UNet_BE, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)

        # self.att_1_8 = BCModule(256)  # +
        self.cbl_1_8 = BELModule(256, 256*5//4)
        self.bce_1_8 = BEBModule(256*5//4, 1)

        self.up2 = Up(256, 128)
        # self.att_1_4 = BCModule(128)  # +
        self.cbl_1_4 = BELModule(128, 128*5//2)
        self.bce_1_4 = BEBModule(128*5//2, 1)

        self.up3 = Up(128, 64)
        # self.att_1_2 = BCModule(64)  # +
        self.cbl_1_2 = BELModule(64, 64*5//2)
        self.bce_1_2 = BEBModule(64*5//2, 1)

        self.up4 = Up(64, 32)
        self.outc = OutConv(32, self.num_classes) 


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        # x = self.att_1_8(x)  # +
        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x = bce_1_8*(x+1)

        x = self.up2(x, x3)
        # x = self.att_1_4(x)  # +
        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x = bce_1_4*(x+1)

        x = self.up3(x, x2)
        # x = self.att_1_2(x)  # +
        cbl_1_2 = self.cbl_1_2(x)
        bce_1_2 = self.bce_1_2(cbl_1_2)
        x = bce_1_2*(x+1)

        x = self.up4(x, x1)
        output = self.outc(x)

        cbl = {'cbl_1_2': cbl_1_2, 'cbl_1_4': cbl_1_4, 'cbl_1_8': cbl_1_8}
        bce = {'bce_1_2': bce_1_2, 'bce_1_4': bce_1_4, 'bce_1_8': bce_1_8}

        return output, cbl, bce


class UNet_BE_1(nn.Module):
    def __init__(self, configer):
        super(UNet_BE_1, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.cbl_1_8 = BELModule(256, 256*5//4)
        self.bce_1_8 = OutConv(256*5//4, 1)

        self.up2 = Up(256, 128)
        self.cbl_1_4 = BELModule(128, 128*5//2)
        self.bce_1_4 = OutConv(128*5//2, 1)

        self.up3 = Up(128, 64)
        self.cbl_1_2 = BELModule(64, 64*5//2)
        self.bce_1_2 = OutConv(64*5//2, 1)

        self.up4 = Up(64, 32)
        self.cbl_1_1 = BELModule(32, 32*5//2)
        self.bce_1_1 = OutConv(32*5//2, 1)
        # self.cbl_1_1 = BELModule(32, 120)
        # self.bce_1_1 = OutConv(120, 1)

        self.outc = OutConv(32, self.num_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)

        x = self.up2(x*(bce_1_8+1), x3)
        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)

        x = self.up3(x*(bce_1_4+1), x2)
        cbl_1_2 = self.cbl_1_2(x)
        bce_1_2 = self.bce_1_2(cbl_1_2)

        x = self.up4(x*(bce_1_2+1), x1)
        cbl_1_1 = self.cbl_1_1(x)
        bce_1_1 = self.bce_1_1(cbl_1_1)

        output = self.outc(x*(bce_1_1+1))

        cbl = {'cbl_1_1': cbl_1_1, 'cbl_1_2': cbl_1_2, 'cbl_1_4': cbl_1_4, 'cbl_1_8': cbl_1_8}
        bce = {'bce_1_1': bce_1_1, 'bce_1_2': bce_1_2, 'bce_1_4': bce_1_4, 'bce_1_8': bce_1_8}

        return output, cbl, bce
    

class UNet_BC(nn.Module):
    def __init__(self, configer):
        super(UNet_BC, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.att_1_8 = BCModule(256)

        self.up2 = Up(256, 128)
        self.att_1_4 = BCModule(128)

        self.up3 = Up(128, 64)
        self.att_1_2 = BCModule(64)

        self.up4 = Up(64, 32)
        self.outc = OutConv(32, self.num_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x, con_1_8 = self.att_1_8(x)

        x = self.up2(x, x3)
        x, con_1_4 = self.att_1_4(x)

        x = self.up3(x, x2)
        x, con_1_2 = self.att_1_2(x)

        x = self.up4(x, x1)
        output = self.outc(x)

        con = {'con_1_2': con_1_2, 'con_1_4': con_1_4, 'con_1_8': con_1_8}

        return output, con


class UNet_BC_1(nn.Module):
    def __init__(self, configer):
        super(UNet_BC_1, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.att_1_8 = BCModule(256)

        self.up2 = Up(256, 128)
        self.att_1_4 = BCModule(128)

        self.up3 = Up(128, 64)
        self.att_1_2 = BCModule(64)

        self.up4 = Up(64, 32)
        self.att_1_1 = BCModule(32)

        self.outc = OutConv(32, self.num_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x, con_1_8 = self.att_1_8(x)

        x = self.up2(x, x3)
        x, con_1_4 = self.att_1_4(x)

        x = self.up3(x, x2)
        x, con_1_2 = self.att_1_2(x)

        x = self.up4(x, x1)
        x, con_1_1 = self.att_1_1(x)

        output = self.outc(x)

        con = {'con_1_1': con_1_1, 'con_1_2': con_1_2, 'con_1_4': con_1_4, 'con_1_8': con_1_8}

        return output, con


class UNet_BE_BC(nn.Module):
    def __init__(self, configer):
        super(UNet_BE_BC, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.att_1_8 = BCModule(256)
        self.cbl_1_8 = BELModule(256, 256*5//4)
        self.bce_1_8 = BEBModule(256*5//4, 1)

        self.up2 = Up(256, 128)
        self.att_1_4 = BCModule(128)
        self.cbl_1_4 = BELModule(128, 128*5//2)
        self.bce_1_4 = BEBModule(128*5//2, 1)

        self.up3 = Up(128, 64)
        self.att_1_2 = BCModule(64)
        self.cbl_1_2 = BELModule(64, 64*5//2)
        self.bce_1_2 = BEBModule(64*5//2, 1)

        self.up4 = Up(64, 32)
        self.att_1_1 = BCModule(32)

        # self.outc = OutConv(32, self.num_classes)
        self.outc = OutConv_nosig(32, self.num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.att_1_8(x)
        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x = bce_1_8*(x+1)

        x = self.up2(x, x3)
        x = self.att_1_4(x)
        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x = bce_1_4*(x+1)

        x = self.up3(x, x2)
        x = self.att_1_2(x)
        cbl_1_2 = self.cbl_1_2(x)
        bce_1_2 = self.bce_1_2(cbl_1_2)
        x = bce_1_2*(x+1)
        
        x = self.up4(x, x1)
        x = self.att_1_1(x)
        # output = self.outc(x)
        con = self.outc(x)
        output = self.sigmoid(con)

        cbl = {'cbl_1_2': cbl_1_2, 'cbl_1_4': cbl_1_4, 'cbl_1_8': cbl_1_8}
        bce = {'bce_1_2': bce_1_2, 'bce_1_4': bce_1_4, 'bce_1_8': bce_1_8}

        return output, cbl, bce, con
   


class UNet_BE_BC_1(nn.Module):
    def __init__(self, configer):
        super(UNet_BE_BC_1, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.cbl_1_8 = BELModule(256, 256*5//4)
        self.bce_1_8 = OutConv(256*5//4, 1)
        self.att_1_8 = BCModule(256)

        self.up2 = Up(256, 128)
        self.cbl_1_4 = BELModule(128, 128*5//2)
        self.bce_1_4 = OutConv(128*5//2, 1)
        self.att_1_4 = BCModule(128)

        self.up3 = Up(128, 64)
        self.cbl_1_2 = BELModule(64, 64*5//2)
        self.bce_1_2 = OutConv(64*5//2, 1)
        self.att_1_2 = BCModule(64)

        self.up4 = Up(64, 32)
        self.cbl_1_1 = BELModule(32, 32*5)
        self.bce_1_1 = OutConv(32*5, 1)
        self.att_1_1 = BCModule(32)
        self.outc = OutConv(32, self.num_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        cbl_1_8 = self.cbl_1_8(x)
        bce_1_8 = self.bce_1_8(cbl_1_8)
        x = x*(bce_1_8+1)
        x, con_1_8 = self.att_1_8(x)

        x = self.up2(x, x3)
        cbl_1_4 = self.cbl_1_4(x)
        bce_1_4 = self.bce_1_4(cbl_1_4)
        x = x*(bce_1_4+1)
        x, con_1_4 = self.att_1_4(x)

        x = self.up3(x, x2)
        cbl_1_2 = self.cbl_1_2(x)
        bce_1_2 = self.bce_1_2(cbl_1_2)
        x = x*(bce_1_2+1)
        x, con_1_2 = self.att_1_2(x)

        x = self.up4(x, x1)
        cbl_1_1 = self.cbl_1_1(x)
        bce_1_1 = self.bce_1_1(cbl_1_1)
        x = x*(bce_1_1+1)
        x, con_1_1 = self.att_1_1(x)

        output = self.outc(x)

        cbl = {'cbl_1_1': cbl_1_1, 'cbl_1_2': cbl_1_2, 'cbl_1_4': cbl_1_4, 'cbl_1_8': cbl_1_8}
        bce = {'bce_1_1': bce_1_1, 'bce_1_2': bce_1_2, 'bce_1_4': bce_1_4, 'bce_1_8': bce_1_8}
        con = {'con_1_1': con_1_1, 'con_1_2': con_1_2, 'con_1_4': con_1_4, 'con_1_8': con_1_8}

        return output, cbl, bce, con
