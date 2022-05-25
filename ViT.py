import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, IE, SAHead1, SAHead2, SAHead3, SAHead4, SAHead5,
                 SA, CB, Cls, Encoder, head_num, img_channels, out_channels,
                 cls_num, fp_num):
        super().__init__()

        self.ImageEmbed = IE(img_channels, out_channels)
        # out size: b, 11, 4, 120, 120

        # totally 10 Encoder blocks
        # 1 block
        self.Encoder0 = Encoder(SA, CB, SAHead1, head_num, 4, fp_num, 1, 1)
        # output size: b, 11, 4, 120, 120

        # 2 block
        self.Encoder1 = Encoder(SA, CB, SAHead1, head_num, 4, fp_num, 1, 1)
        # output size: b, 11, 4, 120, 120

        # 3 block
        self.Encoder2 = Encoder(SA, CB, SAHead1, head_num, 4, fp_num, 2, 4)
        # output size: b, 11, 16, 60, 60

        # 4 block
        self.Encoder3 = Encoder(SA, CB, SAHead2, head_num, 16, fp_num, 1, 1)
        # output size: b, 11, 16, 60, 60

        # 5 block
        self.Encoder4 = Encoder(SA, CB, SAHead2, head_num, 16, fp_num, 2, 1)
        # output size: b, 11, 16, 30, 30

        # 6 block
        self.Encoder5 = Encoder(SA, CB, SAHead3, head_num, 16, fp_num, 1, 1)
        # output size: b, 11, 16, 30, 30

        # 7 block
        self.Encoder6 = Encoder(SA, CB, SAHead3, head_num, 16, fp_num, 2, 4)
        # output size: b, 11, 64, 15, 15

        # 8 block
        self.Encoder7 = Encoder(SA, CB, SAHead4, head_num, 64, fp_num, 1, 1)
        # output size: b, 11, 64, 15, 15

        # 9 block
        self.Encoder8 = Encoder(SA, CB, SAHead4, head_num, 64, fp_num, 2, 1)
        # output size: b, 11, 64, 8, 8

        # 10 block
        self.Encoder9 = Encoder(SA, CB, SAHead5, head_num, 64, fp_num, 2, 2)
        # output size: b, 11, 128, 4, 4

        self.ClsHead = Cls(128, cls_num, fp_num)

    def forward(self, x):
        x = self.ImageEmbed(x)

        x = self.Encoder0(x)
        x = self.Encoder1(x)
        x = self.Encoder2(x)
        x = self.Encoder3(x)
        x = self.Encoder4(x)
        x = self.Encoder5(x)
        x = self.Encoder6(x)
        x = self.Encoder7(x)
        x = self.Encoder8(x)
        x = self.Encoder9(x)

        x = self.ClsHead(x)

        return x


