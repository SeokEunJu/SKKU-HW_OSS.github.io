import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import resnet50
import cv2
import os

from utils import Dataset
from utils import normalization

from model import Generator, Discriminator

proj_directory = '/project'
data_directory = '/dataset'
validation_directory = ''

celeba = os.path.join(data_directory, 'img_align_celeba')
menpo = os.path.join(data_directory, 'LS3D-W/Menpo-3D')
_300w = os.path.join(data_directory, 'LS3D-W/300W-Testset-3D')
aflw = os.path.join(data_directory, 'LS3D-W/AFLW2000-3D-Reannotated')

save_path_G = os.path.join(proj_directory, 'generator.pth')
save_path_D = os.path.join(proj_directory, 'discriminator.pth')

batch_size = 8
resnet = resnet50(pretrained=True).eval()

def train(train_directories, n_epoch):
    print('start')

    dataset = Dataset(train_directories)
    loaded_training_data = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )

    valid_dataset = Dataset(validation_directory)  # evaluation dataset
    loaded_valid_data = DataLoader(dataset=valid_dataset, batch_size=1)

    generator = Generator()  # .cuda()
    discriminator = Discriminator()  # .cuda()

    if os.path.exists(save_path_G):
        generator.load_state_dict(torch.load(save_path_G))
        print('reading generator checkpoints...')
    if os.path.exists(save_path_D):
        discriminator.load_state_dict(torch.load(save_path_D))
        print('reading discriminator checkpoints...')
    if not os.path.exists(os.path.join(proj_directory, 'validation')):
        os.makedirs(os.path.join(proj_directory, 'validation'))

    mse = nn.MSELoss()
    resnet_feature = nn.Sequential(*list(vgg.features)[:-1])

    learning_rate = 2.5e-4
    final_lr = 1e-5
    decay = (learning_rate - final_lr) / n_epoch

    print('train with MSE and perceptual loss')
    for epoch in range(n_epoch):
        G_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
        for i, data in enumerate(loaded_training_data):
            lr, gt, img_name = data
            gt = gt.float()
            # lr = lr.cuda()
            # gt = gt.cuda()

            # checking dataloader - step 1
            # if there is a pair of outputs LR and HR at project file, success.
            ########################################################################################
            # lr = lr[0].detach().numpy().transpose(1, 2, 0)                                       #
            # lr = normalization(lr, _from=(0, 1))                                                 #

            # gt = gt[0].detach().numpy().transpose(1, 2, 0)                                       #
            # gt = normalization(gt, _from=(0, 1))                                                 #

            # filename_lr = os.path.join(proj_directory, 'LR_' + img_name[0])                      #
            # filename_hr = os.path.join(proj_directory, 'HR_' + img_name[0])                      #
            # cv2.imwrite(filename_lr, lr)                                                         #
            # cv2.imwrite(filename_hr, gt)                                                         #
            # quit()                                                                               #
            ########################################################################################

            # forwarding
            sr = generator(lr)

            # checking network formation - step2
            # if the output of the print statement above is 'torch.Size([4, 3, 64, 64])', success
            ########################################################################################
            # print(sr.shape)                                                                      #
            # quit()                                                                               #
            ########################################################################################

            # initialization
            G_optimizer.zero_grad()

            # loss computation
            mse_loss = mse(sr, gt)

            g_loss = mse_loss # + 0.006 * perceptual_loss

            # g_loss.backward(retain_graph=True)
            g_loss.backward()
            G_optimizer.step()

            if i % 10 == 0:
                print("loss at %d : %d ==>\t%.6f (%.6f + %.6f)" % (epoch, i, g_loss, mse_loss, vgg_loss))

        if epoch % 2 == 0:
            validation = os.path.join(proj_directory, 'validation', str(epoch))
            os.makedirs(validation)
            for _, val_data in enumerate(loaded_valid_data):
                lr, _, img_name = val_data
                sr = generator(lr)
                sr = sr[0]
                sr = normalization(sr)
                sr = sr.cpu().detach().numpy().transpose(1, 2, 0)
                img_name = img_name[0]

                filename = os.path.join(validation, img_name + '.png')
                cv2.imwrite(filename=filename, img=sr)

            # save checkpoints
            torch.save(generator.state_dict(), save_path_G)
            # torch.save(discriminator.state_dict(), save_path_D)

        # decay learning rate after one epoch
        learning_rate -= decay


    # save checkpoints
    torch.save(generator.state_dict(), save_path_G)
    # torch.save(discriminator.state_dict(), save_path_D)
    print('training finished.')


if __name__ == '__main__':
    epochs = 70
    train_directories = []
    train(train_directories, epochs)
