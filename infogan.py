import os
import time
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Conv2D, Dropout, \
        Flatten, Reshape, ReLU, Input, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation
from tensorflow.python.data import Iterator
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import scipy as sp
import sys
import multiprocessing
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tfd = tfp.distributions


def make_directory(PATH):
    if not os.path.exists(PATH):
            os.mkdir(PATH)  

class InfoGAN(object):
    def __init__(self, opt):
        self.opt = opt
        self.loss_type = opt.loss_type
        self.alpha = opt.alpha
        if self.loss_type == 'vanilla':
            self.alpha = 1.0
        self.c_type = opt.c_type
        self.seed = opt.seed
        self.batch_size = 128
        self.z_dim = 62
        self.num_classes = 10
        self.c_dim1 = 10
        self.c_dim2 = 1
        self.c_dim3 = 1
        self.dataset = opt.dataset
        if self.c_type == 'discrete':
            self.c_dim2, self.c_dim3 = 0, 0
        self.noise_dim = self.z_dim + self.c_dim1 + self.c_dim2 + self.c_dim3
        self.lambda_c = opt.lambda_c
        self.lambda_d = opt.lambda_d
        self.n_epochs = opt.n_epochs
        self.scores = np.zeros(self.n_epochs)
        self.intrafids = np.zeros((self.n_epochs, 10))
        self.epsilon = 1e-8
        self.d_opt = Adam(2e-4, beta_1=0.5)
        self.g_opt = Adam(1e-3, beta_1=0.5)
        self.q_opt = Adam(2e-4, beta_1=0.5)
        self.num_img_batch = 4

        if self.c_type == 'discrete':
            self.num_img_batch = 500
        self.desc = '\n'.join([f'Loss Type: {self.loss_type}', self.c_type, f'Alpha: {self.alpha}', 
                                f'Seed:{self.seed}', f'Dataset: {self.dataset}'])
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

    def get_data(self):
        (self.train_img, _), (self.test_img, _) = tf.keras.datasets.mnist.load_data()
  
        if self.dataset == 'fashion':
            (self.train_img, self.train_labels), (self.test_img, _) = tf.keras.datasets.fashion_mnist.load_data()
        if self.dataset == 'cifar10':
            (self.train_img, _), (self.test_img, _) = tf.keras.datasets.cifar10.load_data()
            self.train_img = self.train_img.reshape(self.train_img.shape[0], 32, 32, 3)
            self.test_img = self.test_img.reshape(self.test_img.shape[0], 32, 32, 3)
        if self.dataset == 'fashion' or self.dataset == 'mnist':
            self.train_img = self.train_img.reshape(self.train_img.shape[0], 28, 28, 1)
            self.test_img = self.test_img.reshape(self.test_img.shape[0], 28, 28, 1)
        self.real_mu, self.real_sigma = self.get_eval_metrics(self.train_img)
        self.train_data, self.test_data = self.clean_data(self.train_img, train = True), self.clean_data(self.test_img, train = False)


    def get_eval_metrics(self, data):
        img_dims = data.shape
        eval_img = data[np.random.choice(img_dims[0], 10000, replace=False), :, :, :]
        eval_img = eval_img.reshape(10000, np.prod(img_dims[1:])).astype('float32')
        eval_img = eval_img / 255.0
        real_mu = eval_img.mean(axis = 0)
        eval_img = np.transpose(eval_img)
        real_sigma = np.cov(eval_img)
        return real_mu, real_sigma

    def clean_data(self, data, train):
        
        new_data = data.astype('float32')
        new_data = (new_data - 127.5) / 127.5
        if train:
            new_data = tf.data.Dataset.from_tensor_slices(new_data)
            return new_data.shuffle(60000).batch(self.batch_size)
        return new_data


    def gen_loss_vanilla(self):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        return bce(tf.ones_like(self.fake_predicted_labels), self.fake_predicted_labels) + self.info_loss()
    
    def gen_loss_vanilla_l1(self):
        return tf.math.abs(self.gen_loss_vanilla() - (-tf.math.log(2.0)))  + self.info_loss()

    def gen_loss_renyi(self):
        f = tf.math.reduce_mean(tf.math.pow(1 - self.fake_predicted_labels,
                                                (self.alpha - 1) * tf.ones_like(self.fake_predicted_labels)))
        gen_loss = tf.math.abs(1.0 / (self.alpha - 1) * tf.math.log(f + self.epsilon) + tf.math.log(2.0))
        return gen_loss
        #return gen_loss + self.info_loss()

    def dis_loss(self):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        real_loss = bce(tf.ones_like(self.real_predicted_labels), self.real_predicted_labels)
        fake_loss = bce(tf.zeros_like(self.fake_predicted_labels), self.fake_predicted_labels)
        return real_loss + fake_loss

    def info_loss(self):
        c_entropy_d = K.mean(- K.sum(K.log(self.c_pred_d + self.epsilon) * self.c_d, axis=1))
        entropy_d = K.mean(- K.sum(K.log(self.c_d+ self.epsilon) * self.c_d, axis=1))
        if self.c_type != 'discrete':
            c_entropy_c =  K.mean(- K.sum(K.log(self.c_pred_c + self.epsilon) * self.c_c, axis=1))
            #c_entropy_c =  K.mean(- K.sum((self.c_pred_c + self.epsilon) * self.c_c, axis=1))
            entropy_c = K.mean(- K.sum(K.log(self.c_c+ self.epsilon) * self.c_c, axis=1))
            return self.lambda_d*(c_entropy_d  + entropy_d) + self.lambda_c*(c_entropy_c+entropy_c)

        return c_entropy_d


    def build_generator(self):
        z = Input(shape = (self.z_dim,), name = 'z')
        c1 = Input(shape = (self.c_dim1,), name = 'c1')
        c2 = Input(shape = (self.c_dim2,), name = 'c2')
        c3 = Input(shape = (self.c_dim3,), name = 'c3')
        
        concat_layer = Concatenate(axis = 1)([z, c1, c2, c3])
        if self.c_type == 'discrete':
            concat_layer = Concatenate(axis = 1)([z, c1])
        fc1 = Dense(1024)(concat_layer)
        rl1 = ReLU()(fc1)
        bn1 = BatchNormalization()(rl1)
        fc2 = Dense(7*7*128)(bn1)
        rs = Reshape((7, 7, 128))(fc2)
        rl2 = ReLU()(rs)
        bn2 = BatchNormalization()(rl2)
        conv_layer1 = Conv2DTranspose(64, (4, 4), strides = 2, padding = 'same')(bn2)
        rl3 = ReLU(name = 'ReLU3')(conv_layer1)
        bn3 = BatchNormalization()(rl3)
        conv_layer2 = Conv2DTranspose(1, (4, 4), strides = 2, padding = 'same')(bn3)
        
        model = Model([z, c1, c2, c3], conv_layer2)
        if self.c_type == 'discrete':
            model = Model([z, c1], conv_layer2)
        return model

    def build_dq(self):
       
        image_input = Input(shape = (28, 28, 1))
        conv_layer1 = Conv2D(64, (4, 4), strides = 2, padding = 'same')(image_input)
        lrelu1 = LeakyReLU(alpha = 0.1)(conv_layer1)
        conv_layer2 = Conv2D(128, (4, 4), strides = 2, padding = 'same')(lrelu1)
        bn1 = BatchNormalization()(conv_layer2)
        fc1 = Dense(1024)(bn1)
        lrelu2 = LeakyReLU(alpha = 0.1)(fc1)
        bn2 = BatchNormalization()(lrelu2)
        flat = Flatten()(bn2)

        discriminator_output = Dense(1, activation='sigmoid')(flat)
        fc_q = Dense(128)(flat)
        relu_q = ReLU()(fc_q)
        bn_q = BatchNormalization()(relu_q)
        c1_pred = Dense(self.num_classes, activation='softmax')(bn_q)
        mu_c2 = Dense(1)(bn_q)
        sigma_c2 = Dense(1, activation = lambda x:tf.math.exp(x))(bn_q)
        mu_c3 = Dense(1)(bn_q)
        sigma_c3 = Dense(1, activation = lambda x:tf.math.exp(x))(bn_q)

        discriminator = Model(inputs = image_input, outputs=discriminator_output)
        q_model = Model(image_input, [c1_pred, mu_c2, sigma_c2, mu_c3, sigma_c3])

        if self.c_type == 'discrete':
            q_model = Model(image_input, c1_pred)

        return discriminator, q_model

    @tf.function
    def train_step(self, real_images):
        
        z = tf.random.normal([self.batch_size, self.z_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape, tf.GradientTape() as q_tape:
            self.img = real_images
            self.discriminator.trainable = True
            c1 = tfd.Categorical(probs=tf.ones([self.num_classes])*(1.0/self.num_classes)).sample((self.batch_size,))
            self.c_d = tf.one_hot(c1, self.num_classes)
            c2 = tf.random.uniform([self.batch_size, self.c_dim2], minval=-1, maxval=1)
            c3 = tf.random.uniform([self.batch_size, self.c_dim3], minval=-1, maxval=1)
            self.c_c = tf.concat([c2, c3], axis = 1)
            self.real_predicted_labels = self.discriminator(real_images, training = True)
            if self.c_type != 'discrete':
                self.generated_images = self.generator([z, self.c_d, c2, c3], training = True)
            else:
                self.generated_images = self.generator([z, self.c_d], training = True)
            self.fake_predicted_labels = self.discriminator(self.generated_images, training = True)
            if self.c_type != 'discrete':
                c1_pred, mu1, sigma1, mu2, sigma2 = self.q_model(self.generated_images, training = True)
                dist_c2 = tfd.Normal(loc = mu1, scale = sigma1)
                dist_c3 = tfd.Normal(loc = mu2, scale = sigma2)
                c2_pred = dist_c2.prob(c2)
                c3_pred = dist_c3.prob(c3)
                
                self.c_pred_d = c1_pred
                self.c_pred_c = tf.concat([c2_pred, c3_pred], axis = 1)
            else:
                self.c_pred_d = self.q_model(self.generated_images, training = True)
            self.dis_loss_value = self.dis_loss()
            
            self.gen_loss_value = self.generator_loss()
            self.q_loss_value = self.info_loss()


        dis_gradients = dis_tape.gradient(self.dis_loss_value, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))
        self.discriminator.trainable = False
        gen_gradients = gen_tape.gradient(self.gen_loss_value, self.generator.trainable_variables)
        q_gradients = q_tape.gradient(self.q_loss_value, self.q_model.trainable_variables)
        self.g_opt.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.q_opt.apply_gradients(zip(q_gradients, self.q_model.trainable_variables))
        return self.dis_loss_value, self.gen_loss_value, self.q_loss_value

    def build_directory(self):
        make_directory('InfoGAN')
        make_directory(f'InfoGAN/alpha-{self.alpha}')
        subfolders = [f[0] for f in os.walk(f'InfoGAN/alpha-{self.alpha}')]
        folders = [f for f in subfolders if f.startswith(f'InfoGAN/alpha-{self.alpha}/v')]
        versions = [f.split('/v')[1] for f in folders]
        versions = [int(v) for v in versions if v.isnumeric()]
        version = 1
        if versions:
            version = max(versions) + 1
        self.path = f'InfoGAN/alpha-{self.alpha}/v'+str(version)
        make_directory(self.path)
        make_directory(self.path + '/metrics')
        make_directory(self.path + '/metrics/accuracy')
        make_directory(self.path + '/metrics/losses')
        make_directory(self.path + '/img')
        make_directory(self.path + '/models')
 
        with open(self.path+'/description.txt', 'w') as f:
            f.write(f'version={version}\n')
            for k, v in vars(self.opt).items():
                f.write(f'{k}={v}')
                f.write('\n')
    
    def build_gan(self):
        self.generator = self.build_generator()
        self.discriminator, self.q_model = self.build_dq()
        self.generator_loss = self.gen_loss_vanilla
        if self.loss_type == 'renyi':
            self.generator_loss = self.gen_loss_renyi
        if self.loss_type == 'vanilla_l1':
            self.generator_loss = self.gen_loss_vanilla_l1

    def train(self):
        self.get_data()
        self.build_gan()
        self.build_directory()
        gen_loss_history = np.zeros(self.n_epochs)
        dis_loss_history = np.zeros(self.n_epochs)
        q_loss_history = np.zeros(self.n_epochs)
        epoch_times = []
        img_times = []
        for epoch in range(1, self.n_epochs + 1):
            print(f"Epoch {epoch}")
            n_batches = 0
            #self.get_data()
            train_iter = iter(self.train_data)
            start_epoch = time.time()
            for real_images in train_iter:
                dis_loss_value, gen_loss_value, q_loss_value = self.train_step(real_images)
                gen_loss_history[epoch - 1] += gen_loss_value
                dis_loss_history[epoch - 1] += dis_loss_value
                q_loss_history[epoch - 1] += q_loss_value
                n_batches += 1
            gen_loss_history = gen_loss_history/n_batches
            dis_loss_history = dis_loss_history/n_batches
            q_loss_history = q_loss_history/n_batches
            #self.evaluate(epoch)
            end_epoch = time.time()
            epoch_times.append(end_epoch - start_epoch)
            start_img = time.time()
            self.save_generated_images(epoch)
            end_img = time.time()
            img_times.append(end_img - start_img)
        np.save(self.path + '/metrics/losses/gen_loss.npy', gen_loss_history)
        np.save(self.path + '/metrics/losses/dis_loss.npy', dis_loss_history)
        np.save(self.path + '/metrics/losses/q_loss.npy', q_loss_history)
        self.generator.save(self.path+ '/models/generator')
        self.discriminator.save(self.path + '/models/discriminator')
        self.q_model.save(self.path + '/models/q_model')
        time_df = pd.DataFrame({'epoch':list(range(1, self.n_epochs + 1)),
        'epoch_time':epoch_times, 'img_times':img_times})

        time_df.to_pickle(self.path+'/times.pkl')

    
    def evaluate(self, epoch):
        real_images = self.test_data
        c1 = np.random.randint(0, 10, size = 10000).reshape((10000, 1))
        c2 = tf.random.uniform([10000, self.c_dim2], minval=-1, maxval=-1)
        c3 = tf.random.uniform([10000, self.c_dim3], minval=-1, maxval=-1)
        z = tf.random.normal([10000, self.z_dim])
        if self.c_type == 'discrete':
            generated_images = self.generator([z, c1])
        else:
            generated_images = self.generator([z, c1, c2, c3])

        real_pred = np.ravel(self.discriminator(real_images))
        gen_pred = np.ravel(self.discriminator(generated_images))

        predictions = np.concatenate((real_pred, gen_pred))
        predictions[predictions < 0.5] = 0
        predictions[predictions > 0.5] = 1
        trues = np.concatenate((np.ones_like(real_pred), np.zeros_like(gen_pred)))

        confusion_matrix = tf.math.confusion_matrix(trues, predictions)
        
        np.save(self.path + '/metrics/accuracy/confusion_matrix'+str(epoch)+'.npy', confusion_matrix)

    def save_generated_images(self, epoch):
        c1 = np.array([i for i in range(10)])
        if self.c_type != 'discrete':
            if epoch == 1:
                c2, c3 = np.linspace(-2, 2, 20), np.linspace(-2, 2, 20)
                c = np.array(np.meshgrid(c1, c2, c3)).T.reshape(self.num_classes, 20, 20, 3)
                self.c1_scalar = tf.tile(np.ravel(c[:, :, :, 0]).reshape(-1, 1), multiples=[self.num_img_batch, 1])
                self.c1_scalar = np.array(self.c1_scalar, dtype='int32')
                
                self.c2_eval = tf.tile(np.ravel(c[:, :, :, 1]).reshape(-1, 1), multiples=[self.num_img_batch, 1])
                self.c3_eval = tf.tile(np.ravel(c[:, :, :, 2]).reshape(-1, 1), multiples=[self.num_img_batch, 1])
                self.z_eval = tf.random.normal([self.num_classes*20*20*self.num_img_batch, self.z_dim])
                self.c1_eval = tf.one_hot(self.c1_scalar, self.num_classes)
                self.c1_eval = tf.reshape(self.c1_eval, [self.c1_eval.shape[0], self.num_classes])
                c_index = np.hstack((self.c1_scalar, self.c2_eval, self.c3_eval))
                np.save(self.path + '/img/c_index.npy', c_index)
                

            imgs = self.generator([self.z_eval, self.c1_eval, self.c2_eval, self.c3_eval])
            
        else:
            if epoch == 1:
                self.c1_scalar = tf.tile(np.ravel(c1).reshape(-1,1), multiples = [self.num_img_batch, 1])
                self.c1_eval = tf.one_hot(self.c1_scalar, self.num_classes)
                self.c1_eval = tf.reshape(self.c1_eval, [self.c1_eval.shape[0], self.num_classes])
                self.z_eval = tf.random.normal([self.num_classes*self.num_img_batch, self.z_dim])
                c_index = self.c1_scalar
                np.save(self.path + '/img/c_index.npy', c_index)
            imgs = self.generator([self.z_eval, self.c1_eval])

        
        np.save(self.path+'/img/predictions' + str(epoch) + '.npy', imgs)

    def compute_fid(self):
        fake_images = self.generator(tf.random.normal([10000, self.noise_dim]))
        fake_images = fake_images.numpy()
        if self.dataset == 'mnist':
            fake_images = fake_images.reshape(10000, 28*28)
        elif self.dataset == 'cifar10':
            fake_images = fake_images.reshape(10000, 32*32*3)
        fake_images = (fake_images * 127.5 + 127.5) / 255.0
        fake_mu = fake_images.mean(axis=0)
        fake_sigma = np.cov(np.transpose(fake_images))
        covSqrt = sp.linalg.sqrtm(np.matmul(fake_sigma, self.real_sigma))
        if np.iscomplexobj(covSqrt):
            covSqrt = covSqrt.real
        fidScore = np.linalg.norm(self.real_mu - fake_mu) + np.trace(self.real_sigma + fake_sigma - 2 * covSqrt)
        return fidScore
    

    def compute_intrafid(self):
        
