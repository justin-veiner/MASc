from renyigan import *
from infogan import *
from renyi_infogan import *
from alphagan import *
from lkgan import *
import scipy as sp
import multiprocessing
import gc
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gan_type', type=str, default="info")
parser.add_argument('--alpha', type=float, default=3.0)
parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--c_type', type=str, default="discrete")
parser.add_argument('--n_epochs', type=int, default=250)
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--loss_type', type=str, default="vanilla")
parser.add_argument('--lambda_d', type=float, default=1.0)
parser.add_argument('--lambda_c', type=float, default= 0.1)
parser.add_argument('--num_images', type=int, default=10000)
parser.add_argument('--gp', action='store_true')
parser.add_argument('--gen_lr', type=float, default=2e-4)
parser.add_argument('--dis_lr', type=float, default=2e-4)
parser.add_argument('--q_lr', type=float, default=2e-4)
parser.add_argument('--gp_coef', type=float, default=5.0)
parser.add_argument('--alpha_d', type=float, default=3.0)
parser.add_argument('--alpha_g', type=float, default=3.0)
parser.add_argument('--k', type=float, default=2.0)
parser.add_argument('--shifted', action = 'store_true')
parser.add_argument('--l1', action='store_true')

opt = parser.parse_args()


if __name__ == '__main__':
    
    gan = None
    if opt.gan_type == 'renyi':
        gan = RenyiGAN(opt)

    elif opt.gan_type == 'info':
        gan = InfoGAN(opt)

    elif opt.gan_type == 'renyi-info':
        gan = RenyiInfoGAN(opt)
    
    elif opt.gan_type == 'alpha':
        gan = AlphaGAN(opt)
    
    elif opt.gan_type == 'lk':
        gan = LkGAN(opt)

    gan.train()
   
    print("Computing FID Scores:")
    path = gan.path
    del(gan)
    gc.collect()
    if opt.eval:
        compute_fid(path, opt.n_epochs, opt.dataset)
   
