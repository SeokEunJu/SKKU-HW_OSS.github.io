import os
from model import Generator
from utils import *
import torch
from torch.utils.data import DataLoader
from skimage.measure import compare_psnr, compare_ssim

proj_directory = './'
data_directory = '/dataset'
save_path_G = os.path.join(proj_directory, 'ckpt', 'generator.pth')

celeba = os.path.join(data_directory, 'sampled_celeba')

def evaluation(path_to_w, path_to_file):
    generator = Generator().cuda()
    if os.path.exists(path_to_w):
        generator.load_state_dict(torch.load(save_path_G))
        print('reading generator checkpoints...')
    else:
        raise FileNotFoundError('path to Generator weights does not exist')
    if not os.path.exists(os.path.join(proj_directory, 'evaluation')):
        os.makedirs(os.path.join(proj_directory, 'evaluation'))

    evaluation_record = './evaluation.txt'
    record = open(evaluation_record, 'w+')

    dataset = Dataset(path_to_file)
    loaded_valid_dataset = DataLoader(dataset=dataset, batch_size=1)

    total_psnr = 0
    total_ssim = 0

    for i, data in enumerate(loaded_valid_dataset):
        lr, gt, name = data
        sr = generator(lr)
        sr = sr[0]
        sr = normalization(sr, _from=(0, 1))
        sr = sr.cpu().detach().numpy().transpose(1, 2, 0)
        name = name[0]

        gt = gt[0]
        gt = normalization(gt, _from=(0, 1))
        gt = gt.cpu().detach().numpy().transpose(1, 2, 0)

        psnr = compare_psnr(gt, sr)
        ssim = compare_ssim(gt, sr, multichannel=True)

        record.write(name + '\t==>\tPSNR: %.2f\tSSIM: %.2f\n' % (psnr, ssim))

        filename = os.path.join(proj_directory, 'evaluation', name)
        cv2.imwrite(filename=filename, img=sr)

        total_psnr += psnr
        total_ssim += ssim

    avg_psnr = total_psnr / dataset.__len__()
    avg_ssim = total_ssim / dataset.__len__()

    record.write('Average\t==>\tPSNR: %.2f\tSSIM: %.2f\n' % (avg_psnr, avg_ssim))
    record.close()


if __name__ == '__main__':
    eval_directories = [celeba]
    evaluation(save_path_G, eval_directories)
