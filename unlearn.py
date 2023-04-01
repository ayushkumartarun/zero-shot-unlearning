import torch
from torch import nn
from models import View
from torch.nn import functional as F




def attention(x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()


def divergence(student_logits, teacher_logits, KL_temperature):
    divergence = F.kl_div(F.log_softmax(student_logits / KL_temperature, dim=1), F.softmax(teacher_logits / KL_temperature, dim=1))  # forward KL

    return divergence


def KT_loss_generator(student_logits, teacher_logits, KL_temperature):

    divergence_loss = divergence(student_logits, teacher_logits, KL_temperature)
    total_loss = - divergence_loss

    return total_loss


def KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations, KL_temperature = 1, AT_beta = 250):

    divergence_loss = divergence(student_logits, teacher_logits, KL_temperature)
    if AT_beta > 0:
        at_loss = 0
        for i in range(len(student_activations)):
            at_loss = at_loss + AT_beta * attention_diff(student_activations[i], teacher_activations[i])
    else:
        at_loss = 0

    total_loss = divergence_loss + at_loss

    return total_loss

class Generator(nn.Module):

    def __init__(self, z_dim, num_channels = 3):
        super(Generator, self).__init__()
        prefinal_layer = None
        final_layer = None
        if num_channels == 3:
            prefinal_layer = nn.Conv2d(64, 3, 3, stride=1, padding=1)
            final_layer = nn.BatchNorm2d(3, affine=True)
        elif num_channels == 1:
            prefinal_layer = nn.Conv2d(64, 1, 7, stride=1, padding=1)
            final_layer = nn.BatchNorm2d(1, affine=True)
        else:
            print(f"Generator Not Supported for {num_channels} channels")
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 128 * 8**2),
            View((-1, 128, 8, 8)),
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            prefinal_layer,
            final_layer
        )

    def forward(self, z):
        return self.layers(z)

    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print('\n', layer, '---->', act.shape)

            
class LearnableLoader(nn.Module):
    def __init__(self, n_repeat_batch, num_channels = 3,device='cuda'):
        """
        Infinite loader, which contains a learnable generator.
        """

        super(LearnableLoader, self).__init__()
        self.batch_size = 256
        self.n_repeat_batch = n_repeat_batch
        self.z_dim = 128
        self.generator = Generator(self.z_dim, num_channels=num_channels).to(device=device)
        self.device = device

        self._running_repeat_batch_idx = 0
        self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)

    def __next__(self):
        if self._running_repeat_batch_idx == self.n_repeat_batch:
            self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)
            self._running_repeat_batch_idx = 0

        images = self.generator(self.z)
        self._running_repeat_batch_idx += 1
        return images

    def samples(self, n, grid=True):
        """
        :return: if grid returns single grid image, else
        returns n images.
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.z_dim)).to(device=self.device)
            images = visualize(self.generator(z), dataset=self.dataset).cpu()
            if grid:
                images = make_grid(images, nrow=round(math.sqrt(n)), normalize=True)

        self.generator.train()
        return images

    def __iter__(self):
        return self