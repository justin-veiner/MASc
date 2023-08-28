# $\mathcal{L}_\alpha$-GAN experiments
Download [requirements.txt](requirements.txt), [alphagan.py](alphagan.py), [lkgan.py](lkgan.py), [gan_train.py](gan_train.py), and [stacked_mnist.py](stacked_mnist.py) into your project directory.

Navigate to your chosen directory. Create an environment and install the requirements file:
```bash
pip install -r requirements.txt
```

## $\alpha$-GAN
To train an $\alpha$-GAN, run the following command:
```bash
python gan_train.py --gan_type "alpha"
```
Use the following arguments to set additional parameters:
- alpha_d: alpha parameter for the discriminator loss function (positive float, default 3.0)
- alpha_g: alpha parameter for the generator loss function (positive float, default 3.0)
- dataset: dataset to train on ("mnist", "cifar10", "stacked-mnist") (default "mnist")
- seed: seed to use to initialize random numbers (positive integer, default 42)
- n_epochs: number of training epochs (positive integer, default 250)
- num_images: number of sample images produced after each epoch (positive integer, default 10000)
- gen_lr: generator learning rate (positive float, default 2e-4)
- dis_lr: discriminator learning rate (positive float, default 2e-4)
- gp: add a gradient penalty to the discriminator's loss function (bool, default False)
- gp_coef: coefficient for gradient penalty term if added (float, default 5.0)

Note that alpha_d = 1 and alpha_g = 1 runs a DCGAN.

The output folder will have the form ./AlphaGAN/{dataset}/alpha-d{alpha_d}-g{alpha_g}/v{version}. A summary of the parameters used will be in the description.txt file.

## SLkGAN
To train an SLk$GAN, run the following command:
```bash
python gan_train.py --gan_type "lk" --shifted
```
Use the following arguments to set additional parameters:
- k: $k$ parameter for the generator loss function (positive float, default 2.0)
- loss_type: determines the discriminator's loss function ("vanilla" for VanillaGAN, "lk" for LkGAN) (default "vanilla")
- dataset: dataset to train on ("mnist", "cifar10", "stacked-mnist") (default "mnist")
- seed: seed to use to initialize random numbers (positive integer, default 42)
- n_epochs: number of training epochs (positive integer, default 250)
- num_images: number of sample images produced after each epoch (positive integer, default 10000)
- gen_lr: generator learning rate (positive float, default 2e-4)
- dis_lr: discriminator learning rate (positive float, default 2e-4)
- gp: add a gradient penalty to the discriminator's loss function (bool, default False)
- gp_coef: coefficient for gradient penalty term if added (float, default 5.0)

The output folder will have the form ./LkGAN/{loss_type}/{dataset}/k-{k}/v{version}. A summary of the parameters used will be in the description.txt file.

For all $\mathcal{L}_\alpha$-GANs, the FID scores for each epoch will be found in the scores.npy file. The epoch with the best (lowest) FID score will have its generated images saved in the img folder.
