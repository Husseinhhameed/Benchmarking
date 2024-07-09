class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, 1024)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return x

class PoseNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, InceptionV3):
        super(PoseNet, self).__init__()
        self.base_model = InceptionV3

        self.Conv2d_1a_3x3 = InceptionV3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = InceptionV3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = InceptionV3.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = InceptionV3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = InceptionV3.Conv2d_4a_3x3
        self.Mixed_5b = InceptionV3.Mixed_5b
        self.Mixed_5c = InceptionV3.Mixed_5c
        self.Mixed_5d = InceptionV3.Mixed_5d
        self.Mixed_6a = InceptionV3.Mixed_6a
        self.Mixed_6b = InceptionV3.Mixed_6b
        self.Mixed_6c = InceptionV3.Mixed_6c
        self.Mixed_6d = InceptionV3.Mixed_6d
        self.Mixed_6e = InceptionV3.Mixed_6e
        self.Mixed_7a = InceptionV3.Mixed_7a
        self.Mixed_7b = InceptionV3.Mixed_7b
        self.Mixed_7c = InceptionV3.Mixed_7c

        # Out 2
        self.pos2 = nn.Linear(2048, 3, bias=False)
        self.ori2 = nn.Linear(2048, 4, bias=False)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        pos = self.pos2(x)
        ori = self.ori2(x)

        return pos, ori

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated

if __name__ == '__main__':
    dataset_dir = '/content/drive/MyDrive/Dataset'
    img_width, img_height = 256, 256
    batch_size = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = DatasetGenerator(dataset_dir, img_width, img_height, batch_size)
    print(device)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    base_model = models.inception_v3(pretrained=True)
    base_model.aux_logits = False
    model = PoseNet(base_model)

    model = model.to(device)

    inputs, poses = next(iter(data_loader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, 'sample image')

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=50)

    num_epochs = 150

    # Setup for tensorboard



    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for i, (inputs, poses) in enumerate(data_loader):
                inputs = inputs.to(device)
                poses = poses.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pos_out, ori_out = model(inputs)

                    pos_true = poses[:, :3]
                    ori_true = poses[:, 3:]

                    beta = 500
                    ori_out = F.normalize(ori_out, p=2, dim=1)
                    ori_true = F.normalize(ori_true, p=2, dim=1)

                    loss_pos = F.mse_loss(pos_out, pos_true)
                    loss_ori = F.mse_loss(ori_out, ori_true)

                    loss = loss_pos + beta * loss_ori

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loss_print = loss.item()
                loss_ori_print = loss_ori.item()
                loss_pos_print = loss_pos.item()


                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(data_loader.dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        save_filename = 'models/epoch_{}_net.pth'.format(epoch)
        torch.save(model.cpu().state_dict(), save_filename)
        model.to(device)
