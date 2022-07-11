import os
import time
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Flatten, Reshape
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
import utils
from tensorflow.python.data import Iterator
import numpy as np
import traceback
from tensorflow.keras.datasets.cifar10 import load_data
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def make_directory(PATH):
    if not os.path.exists(PATH):
            os.mkdir(PATH)  

def alpha_fn(y1, y2, alpha):
        return tf.math.pow(y1, alpha)*tf.math.pow(y2, 1-alpha)



class GAN(object):
    def __init__(self, dataset, num_trials, gen_loss_type, dis_loss_type, alpha1 = 1.0, alpha2 = 1.0):
        self.batch_size = 32
        self.noise_dim = 100
        self.num_images = 50000
        self.dataset = dataset
        self.num_trials = int(num_trials)
        self.gen_loss_type = gen_loss_type
        self.dis_loss_type = dis_loss_type
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.epsilon =  1e-8
        self.gen_losses = []
        self.dis_losses = []
        self.d_opt = Adam(2e-4, beta_1 = 0.5)
        self.g_opt = Adam(2e-4, beta_1 = 0.5)
        today = datetime.now()
        self.today_date = today.strftime("%b-%d-%Y-%H%M")
        self.gen_loss_renyi, self.dis_loss_rgan, self.gen_loss_rgan, self.dis_loss_renyi, self.dis_loss_vanilla, self.gen_loss_vanilla = self.get_loss_fns()

    
    def get_data(self):
        if self.dataset == 'cifar10':
            (self.train_data, _), (_, _) = tf.keras.datasets.cifar10.load_data()
            self.train_data = self.train_data.reshape(self.train_data.shape[0], 32, 32, 3)
        if self.dataset == 'mnist':
            (self.train_data, _), (_, _) = tf.keras.datasets.mnist.load_data()
            self.train_data = self.train_data.reshape(self.train_data.shape[0], 28, 28, 1)
            self.train_data = self.train_data.reshape
        self.train_data = self.train_data.astype('float32')
        self.train_data = (self.train_data - 127.5) / 127.5
        self.train_data = tf.data.Dataset.from_tensor_slices(self.train_data)
        self.train_data = self.train_data.shuffle(60000).batch(self.batch_size).repeat()
    
    def get_loss_fns(self):
        def gen_loss_renyi(y_true, y_pred):
            f = tf.math.reduce_mean(tf.math.pow(1 - y_pred,
                                                    (self.alpha1- 1) * y_true))
            gen_loss = 1.0 / (self.alpha1 - 1) * tf.math.log(f + self.epsilon)
            return gen_loss
        
        def dis_loss_rgan(y_true, y_pred):
            numerator = alpha_fn(y_true, y_pred, self.alpha2) + alpha_fn(1-y_true, 1-y_pred, self.alpha2)
            denom =  tf.math.pow(y_true, self.alpha2) + tf.math.pow(1-y_true, self.alpha2)

            f = numerator/denom
            return 1.0/(self.alpha2 - 1)*tf.math.reduce_mean(tf.math.log(f + self.epsilon)) + self.dis_loss1

        def gen_loss_rgan(y_true, y_pred):
            numerator = alpha_fn(y_true, y_pred, self.alpha2) + alpha_fn(1-y_true, 1-y_pred, self.alpha2)
            denom = tf.math.pow(y_true, self.alpha2) + tf.math.pow(1-y_true, self.alpha2)

            f = numerator/denom
            return 1.0/(self.alpha2 - 1)*tf.math.reduce_mean(tf.math.log(f + self.epsilon))
        
        def dis_loss_renyi(y_true, y_pred):
        
            real_loss = tf.math.reduce_mean(tf.math.pow(y_true, (self.alpha1 - 1)
                                                        * tf.ones_like(y_true)))
            real_loss = 1.0 / (self.alpha1 - 1) * tf.math.log(real_loss + self.epsilon) + tf.math.log(2.0)
            f = tf.math.reduce_mean(tf.math.pow(1 - y_pred,
                                                (self.alpha1 - 1) * tf.ones_like(y_pred)))
            gen_loss = 1.0 / (self.alpha1 - 1) * tf.math.log(f + self.epsilon) + tf.math.log(2.0)
            dis_loss = - real_loss - gen_loss
            return self.dis_loss1 + dis_loss

        def dis_loss_vanilla(y_true, y_pred):
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            return self.dis_loss1 + bce(y_true, y_pred)

        def gen_loss_vanilla(y_true, y_pred):
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            return bce(y_true, y_pred)

        return gen_loss_renyi, dis_loss_rgan, gen_loss_rgan, dis_loss_renyi, dis_loss_vanilla, gen_loss_vanilla

    
    


    def build_generator(self):
    
        model = Sequential()
    
        model.add(Dense(256*4*4, input_shape=(self.noise_dim,)))
        model.add(LeakyReLU())

        model.add(Reshape((4, 4, 256)))


        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
        model.add(LeakyReLU())


        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
        model.add(LeakyReLU())

        model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
        assert model.output_shape == (None, 32, 32, 3)

        return model

    def build_discriminator(self):
       
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())

        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())

        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())

        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        #model.compile(loss = dis_loss, optimizer=d_opt, metrics=['accuracy'])

        return model

    
    def build_gan(self):
        gen_alpha = '1.0'
        dis_alpha = '1.0'
        self.gen_loss = self.gen_loss_vanilla
        self.dis_loss = self.dis_loss_vanilla
        if self.gen_loss_type == 'renyi':
            self.gen_loss = self.gen_loss_renyi
            gen_alpha = str(self.alpha1)
        if self.dis_loss_type == 'renyi':
            self.dis_loss = self.dis_loss_renyi
            dis_alpha = str(self.alpha1)
        if self.dis_loss_type == 'rgan':
            self.dis_loss = self.dis_loss_rgan
            dis_alpha = str(self.alpha2)
        if self.gen_loss_type == 'rgan':
            self.gen_loss = self.gen_loss_rgan
            gen_alpha = str(self.alpha2)
        

        test_string = 'gan-gen-{genloss}-{genalpha}-dis-{disloss}-{disalpha}'.format(
            genloss=self.gen_loss_type, disloss = self.dis_loss_type, genalpha = gen_alpha, disalpha = dis_alpha)

        subfolders = [ f.name for f in os.scandir('data/'+self.dataset) if f.is_dir() ]

        folders = [folder for folder in subfolders if folder.startswith(test_string)]
        versions = [int(folder.split('-')[-1][1:]) for folder in folders]
        version = 1
        if len(versions) > 0:
            version = max(versions) + 1
        
       
        self.path = '{dataset}/gan-gen-{genloss}-{genalpha}-dis-{disloss}-{disalpha}-v{ver}'.format(
            dataset=self.dataset, genloss=self.gen_loss_type, disloss = self.dis_loss_type, genalpha = gen_alpha, disalpha = dis_alpha,
            ver = version)
       

        make_directory('losses')
        make_directory('losses/'+self.dataset)
        make_directory('losses/'+self.path)
        make_directory('data')
        make_directory('data/'+self.dataset)
        make_directory('data/'+self.path)
        
        
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = self.dis_loss, optimizer = self.d_opt, metrics=['accuracy'])
        self.generator = self.build_generator()
        self.discriminator.trainable = False
        model = Sequential()
        
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(loss = self.gen_loss, optimizer = self.g_opt)

        return model



    def get_real_data(self, train_data, n_samples):
        idx = np.random.randint(0, train_data.shape[0], size=n_samples)
        images = train_data[idx]
        labels = np.ones((n_samples, 1))
        return images, labels

    def get_fake_data(self,  n_samples):
        z = np.random.randn(self.noise_dim * n_samples)
        z = z.reshape(n_samples, self.noise_dim)
        fake_images = self.generator(z)
        fake_labels = np.zeros((n_samples, 1))
        return fake_images, fake_labels


    def train(self, n_epochs):
        for i in range(1, self.num_trials+1):
            self.train_one_trial(n_epochs, i)

    

    def train_one_trial(self, n_epochs, trial):
        self.gan = self.build_gan()
        self.get_data()
        for epoch in range(n_epochs):
            train_iter = iter(self.train_data)
            total_dis_loss = 0
            total_gen_loss = 0
            n_batches = 0
            try:
                for real_images in train_iter:
                    self.dis_loss1 = tf.constant(0).numpy().astype('float64')
                    real_labels = np.ones((self.batch_size, 1))
                    real_predicted_labels = self.discriminator(real_images)
                    real_predicted_labels = real_predicted_labels.numpy().astype('float64')

                    generated_images, generated_labels = self.get_fake_data(self.batch_size)
                    self.dis_loss1 = self.dis_loss(real_labels, real_predicted_labels)
                    self.dis_loss1 = self.dis_loss1.numpy().astype('float32')
                    dis_loss2, _ = self.discriminator.train_on_batch(generated_images, generated_labels)
                    total_dis_loss += self.dis_loss1 + dis_loss2
                    fake_output = np.random.randn(2*self.noise_dim * self.batch_size)
                    fake_output = fake_output.reshape(2*self.batch_size, self.noise_dim)
                    fake_labels = np.ones((2*self.batch_size, 1))
                    gen_loss = self.gan.train_on_batch(fake_output, fake_labels)
                    total_gen_loss += gen_loss
                    n_batches += 1
            except Exception as e:
                print(str(e))
                pass
            self.save_generated_images(epoch + 1, trial)
            avg_gen_loss = total_gen_loss / n_batches
            avg_dis_loss = total_dis_loss / n_batches
            self.gen_losses.append(avg_gen_loss)
            self.dis_losses.append(avg_dis_loss)
        
        make_directory('losses')
        make_directory('losses/'+self.path)
        make_directory('losses/' + self.path + '/trial' + str(trial))
        np.save('losses/' + self.path + '/trial' + str(trial)+'/gen_losses', self.gen_losses)
        np.save('losses/' + self.path + '/trial' + str(trial)+'/dis_losses', self.dis_losses)


    def save_generated_images(self, epoch, trial):
        temp, _  = self.get_fake_data(self.num_images)
        make_directory('data')
        make_directory('data/' + self.path)
        make_directory('data/' + self.path + '/trial' + str(trial))
        np.save('data/' + self.path + '/trial' + str(trial) + '/predictions' + str(epoch), temp)


gen_loss_type = 'rgan'
dis_loss_type = 'rgan'
gan = GAN('cifar10', 1, gen_loss_type, dis_loss_type, alpha1 = 0.1, alpha2 = 0.1)
gan.train(n_epochs = 1)

