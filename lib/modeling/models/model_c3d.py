import torch
import torch.nn as nn


class C3D(nn.Module):
    """
    C3D network
        input size: 112x112
    """

    def __init__(self, num_class=487, pretrained=False, model_dir=None):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_class)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights(model_dir)

    def forward(self, frames):
        # 3x16x112x112
        h = self.relu(self.conv1(frames)) # 64x16x112x112
        h = self.pool1(h) # 64x16x56x56

        h = self.relu(self.conv2(h)) # 128x16x56x56
        h = self.pool2(h) # 128x8x28x28

        h = self.relu(self.conv3a(h)) # 256x8x28x28
        h = self.relu(self.conv3b(h)) # 256x8x28x28
        h = self.pool3(h) # 256x4x14x14

        h = self.relu(self.conv4a(h)) # 512x4x14x14
        h = self.relu(self.conv4b(h)) # 512x4x14x14
        h = self.pool4(h) # 512x2x7x7

        h = self.relu(self.conv5a(h)) # 512x2x7x7
        h = self.relu(self.conv5b(h)) # 512x2x7x7
        h = self.pool5(h) # 512x1x4x4

        h = h.view(-1, 8192) # 8192
        h = self.relu(self.fc6(h)) # 4096
        h = self.dropout(h)
        h = self.relu(self.fc7(h)) # 4096
        h = self.dropout(h)

        logits = self.fc8(h) # 487
        probs = self.softmax(logits)

        return probs

    def __load_pretrained_weights(self, model_dir):
        """Initialiaze network."""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
             # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",
        }

        p_dict = torch.load(model_dir)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_c3d(pretrained=True):
    model = C3D(pretrained=pretrained,
                model_dir='/PRETRAINED_MODEL_DIR'
    )
    modules = list(model.children())[:-6] # before fc
    model = nn.Sequential(*modules)
    return model