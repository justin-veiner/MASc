from renyigan import *
from infogan import *
from renyi_infogan import *
from alphagan import *
from lkgan import *
import scipy as sp
import multiprocessing
import gc
import argparse

class FID:
    def __init__(self):
        print("Computing FID Scores")

    def fid(self, info1, info2):
        (mu1, cov1) = info1  # p_x
        (mu2, cov2) = info2  # p_g
        covSqrt = sp.linalg.sqrtm(np.matmul(cov1, cov2))
        if np.iscomplexobj(covSqrt):
            covSqrt = covSqrt.real
        fidScore = np.linalg.norm(mu1 - mu2) + np.trace(cov1 + cov2
                                                        - 2 * covSqrt)
        return fidScore

    def __call__(self, info):
        (string1, img2, info1) = info
        mu2 = img2.mean(axis=0)
        cov2 = np.cov(np.transpose(img2))
        score = self.fid(info1, (mu2, cov2))
        # print("For alpha = " + string1 + " the FID value is " + str(score))
        return score


def compute_fid(directory, num_epochs, dataset):
    
    img0 = np.load(directory + '/img/predictions1.npy')
    num_images = img0.shape[0]

    if num_images >= 10000:
        num_images = 10000

    (trainIm, trainL), (_, _) = tf.keras.datasets.mnist.load_data() if dataset == 'mnist' else tf.keras.datasets.cifar10.load_data()
    if dataset == 'mnist':
        trainIm = trainIm.reshape(trainIm.shape[0], 28, 28, 1).astype('float32')
        trainIm = trainIm[np.random.choice(50000, num_images, replace=False), :, :, :]
        trainIm = trainIm.reshape(num_images, 28 * 28).astype('float32')
    if dataset == 'cifar10':
        print("cifar10")
        trainIm = trainIm.reshape(trainIm.shape[0], 32, 32, 3).astype('float32')
        trainIm = trainIm[np.random.choice(50000, num_images, replace=False), :, :, :]
        trainIm = trainIm.reshape(num_images, 32*32*3).astype('float32')
    trainIm = trainIm / 255.0
    print(trainIm.shape)
    mu1 = trainIm.mean(axis=0)
    trainIm = np.transpose(trainIm)
    cov1 = np.cov(trainIm)
    info1 = (mu1, cov1)
    proc = FID()
    pool = multiprocessing.Pool(processes=20)
    
    pFiles = []
    for epoch in range(1, num_epochs + 1):
        print(f"Evaluating Epoch {epoch}")
        p = np.load(directory + '/img/predictions' + str(epoch) + '.npy')
        #p = np.load('RenyiGAN/alpha-' + str(alpha) + '/v' + str(ver) + '/img/predictions' + str(epoch) + '.npy')
        p = p.reshape(p.shape[0], 28, 28, 1).astype('float32') if dataset == 'mnist' else p.reshape(p.shape[0], 32, 32, 3)
        p = p[np.random.choice(p.shape[0], num_images, replace=False), :, :, :]
        p = p.reshape(num_images, 28 * 28).astype('float32') if dataset == 'mnist' else p.reshape(10000, 32*32*3)
        p = (p * 127.5 + 127.5) / 255.0
        if np.isnan(p).any():
            break
        pFiles.append(('sim_ann_epoch' + str(epoch), p, info1))
    score_list = pool.map(proc, pFiles)
    np.save(directory + '/scores.npy', score_list)
    #np.save('RenyiGAN/alpha-' + str(alpha) + '/v' + str(ver) + '/scores.npy', score_list)
    #print(score_list)
    # If you are running low on space, uncomment the below section to automatically delete all prediction.npy files 
    # except the epoch when the best FID scores occur 
    for epoch in range(num_epochs):
        if epoch != np.nanargmin(score_list):
            os.remove(directory + '/img/predictions' + str(epoch + 1) + ".npy")

'''
    def compute_intrafid(directory, num_epochs):
        (trainIm, trainL), (_, _) = tf.keras.datasets.mnist.load_data()
        trainIm = trainIm.reshape(trainIm.shape[0], 28, 28, 1).astype('float32')
        trainIm = trainIm.reshape(50000, 28 * 28).astype('float32')
        trainIm = trainIm / 255.0

        print(trainIm.shape)
        mu1 = trainIm.mean(axis=0)
        trainIm = np.transpose(trainIm)
        cov1 = np.cov(trainIm)
        info1 = (mu1, cov1)
        proc = FID()
        pool = multiprocessing.Pool(processes=32)
        pFiles = []
        c_index = np.load(directory + '/img/c_index.npy')

        for epoch in range(1, num_epochs + 1):
            print(f"Evaluating Epoch {epoch}")
            p = np.load(directory + '/img/predictions' + str(epoch) + '.npy')
            #p = np.load('RenyiGAN/alpha-' + str(alpha) + '/v' + str(ver) + '/img/predictions' + str(epoch) + '.npy')
            p = p.reshape(p.shape[0], 28, 28, 1).astype('float32')
            p = p[np.random.choice(50000, 10000, replace=False), :, :, :]
            p = p.reshape(10000, 28 * 28).astype('float32')
            p = (p * 127.5 + 127.5) / 255.0
            if np.isnan(p).any():
                break
            for i in range(10):
                for j in range(10):
                    train_i = trainIm[np.where(trainL == i)[0]]
                    gen_j = p[np.where(c_index[:, 0] == j)[0]]
                    train_i = train_i[np.random.choice(train_i.shape[0], gen_j.shape[0], replace = False)]
'''
            

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
parser.add_argument('--eval', action='store_true')
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
   