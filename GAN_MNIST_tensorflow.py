# -*- coding: utf-8 -*-

"""
GAN(Generative Adversarial Networks)을 이용한 MNIST 데이터 생성
Reference : https://github.com/TengdaHan/GAN-TensorFlow
Author : woojoung
"""

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# MNIST 데이터 로딩
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist.data", one_hot=True)

# Generator에서 생성된 MNIST 이미지를 8x8 grid로 보여주기 위해 plot 함수 정의
def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i]) # gs[i] : 8x8 grid로 subplot 그려주기 
        plt.axis('off')
        plt.imshow(sample.reshape(28, 28))
    
    return fig

# hyperparameter 설정
num_epoch = 100000 # 에폭 크면 로컬 보다는 GPU로 돌리자. 예제에선 100000
batch_size = 64 # 예제에선 64
num_input = 28*28
num_latent_variable = 100
num_hidden = 128
learning_rate = 0.001

# placeholder 선언
X = tf.placeholder(tf.float32, [None, num_input])           # : input image
z = tf.placeholder(tf.float32, [None, num_latent_variable]) # : imput Latent variable

# Generator 함수에서 사용되는 변수들 설정
# 100 -> 128 -> 784 : layer 통과 할 때마다 image 개수 num_input 수에 맞추기.
with tf. variable_scope('generator'):
    # hidden layer paremeter
    G_W1 = tf.Variable(tf.random_normal(shape=[num_latent_variable, num_hidden], stddev=5e-2))
    G_b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    # output layer parameter
    G_W2 = tf.Variable(tf.random_normal(shape=[num_hidden, num_input], stddev=5e-2))
    G_b2 = tf.Variable(tf.constant(0.1, shape=[num_input]))
    
# Discriminator 함수에서 사용되는 변수들 설정 
# 784 -> 128 -> 1 : Generator에서 생성된 784 image 들을 1개로 만들어 인풋 이미지와 구별
with tf.variable_scope('discriminator'):
    # hidden layer parameter
    D_W1 = tf.Variable(tf.random_normal(shape=[num_input, num_hidden], stddev=5e-2))
    D_b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    # output layer parameter
    D_W2 = tf.Variable(tf.random_normal(shape=[num_hidden, 1], stddev=5e-2))
    D_b2 = tf.Variable(tf.constant(0.1, shape=[1]))
    
# Generator 생성하는 함수 정의 
# Inputs: 
#     X : input Latent Variable
# Output:
#    generated_mnist_image : Generator로부터  생성된 MNIST 이미지
def build_generator(X):
    hidden_layer = tf.nn.relu((tf.matmul(X, G_W1) + G_b1))
    output_layer = tf.matmul(hidden_layer, G_W2) + G_b2
    generated_mnist_image = tf.nn.sigmoid(output_layer) # 왜 output image를 생성하는 layer에서 활성함수를 relu 대신에 sigmoid를 사용하였을까? 
    
    return generated_mnist_image # 생성된 이미지 반환.

# Discriminator를 생성하는 함수를 정의
# Inputs:
#   X : 인풋 이미지
# Output:
#   predicted_value : Discriminator가 판단한 True(1) or Fake(0)
#   logits : sigmoid를 씌우기전의 출력값
def build_discriminator(X):
    hidden_layer = tf.nn.relu((tf.matmul(X, D_W1) + D_b1))
    logits = tf.matmul(hidden_layer, D_W2) + D_b2 # logits : 입력 데이터를 네트워크를 통해 전방향 진행하여 나온 결과물을 logits 이라고 부른다.
    predicted_value = tf.nn.sigmoid(logits)     
    
    return predicted_value, logits # 왜 2개 반환? 

# Generator 선언
G = build_generator(z) # : imput Latent variable

# Discriminator 선언
D_real, D_real_logits = build_discriminator(X) # D(X), X: input image, size 28x28, 784
D_fake, D_fake_logits = build_discriminator(G) # D(G(z)), G: generated_mnist_image, size num_imput, 784

# loss function of Discriminator, Discriminator의 손실 함수 정의.
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))    # log(D(x)). # labels: A Tensor of the same type and shape as logits.
#### real이라고 구별할 때 loss
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))   # log(1-D(G(z)))
#### fake라고 구별할 때 loss
d_loss = d_loss_real + d_loss_fake  # log(D(x)) + log(1-D(G(z)))
#### loss의 합, D의 최종적인 loss

##### labels : logits과 똑같은 type과 shape인데 ones, zeros에 따라 값이 1이냐 0. real은 1, fake는 0
# tf.ones_like(tensor): this operation returns a tensor of the same type and shape as tensor with all elements set to 1.
# tf.zeros_like(tensor): this operation returns a tensor of the same type and shape as tensor with all elements set to zero.
#####

# loss function of Generator, Generator의 손실 함수 정의.
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))  #labels=tf.ones_like 와 labels=tf.zeros_like의 차이는? 
#### generator의 labels은 Discriminator에서 fake지만 1(real)이라고 인식 시켜야 하므로 labels=tf.ones_like(D_fake_logits).  

# 전체 파라미터를 Discriminator와 관련된 파라미터와 Generator와 관련된 파라미터로 나눕니다.
tvar = tf.trainable_variables() # tensorflow graph단에 new variable을 add.
dvar = [var for var in tvar if 'discriminator' in var.name]
gvar = [var for var in tvar if 'generator' in var.name]
#####
# tf.trainable_variables() : returns a list of Variable objects.
# Returns all variables created with trainable=True.
# When passed trainable=True, the Variable() constructor automatically adds new variables 
# to the graph collection GraphKeys.TRAINABLE_VARIABLES. 
# This convenience function returns the contents of that collection.
#####

# Discriminator와 Generator의 Optimizer를 정의
d_train_step = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=dvar)
g_train_step = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=gvar)

# 생성된 이미지들을 저장할 generated_outputs 폴더를 생성합니다. 
num_img = 0
if not os.path.exists('generated_output/'):
    os.makedirs('generated_output/')

#####
# os.path.exists(path)
# Return True if path refers to an existing path or an open file descriptor. 
# Returns False for broken symbolic links. 
# On some platforms, this function may return False if permission is not granted to execute os.stat() on the requested file, even if the path physically exists.
#####

with tf.Session() as sess:
    # 변수들에 초기값을 할당한다. 
    sess.run(tf.global_variables_initializer())
    
    # num_epoch 횟수만큼 최적화를 수행한다. 
    for i in range(num_epoch):
        # MNIST 이미지를 batch_size 만큼 불러온다.
        batch_X, _ = mnist.train.next_batch(batch_size)
        # Latent Variable의 인풋으로 사용할 noise를 Uniform Distribution에서 batch_size만큼 샘플링한다.
        batch_noise = np.random.uniform(-1., 1., [batch_size, 100])
        
        # 500번 반복할때마다 생성된 이미지를 저장한다.
        if i % 500 == 0:
            samples = sess.run(G, feed_dict={z: np.random.uniform(-1., 1., [64, 100])}) # z: latent variable의 placeholder
            fig = plot(samples)
            plt.savefig('generated_output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
            num_img += 1
            plt.close(fig)
        
        # Discriminator 최적화를 수행하고 Discriminator의 손실함수를 return한다. 
        _, d_loss_print = sess.run([d_train_step, d_loss], feed_dict={X: batch_X, z: batch_noise})
        
        # Generator 최적화를 수행하고 Generator 손실함수를 return한다.
        _, g_loss_print = sess.run([g_train_step, g_loss], feed_dict={z: batch_noise})
            
        # 100번 반복할때마다 Discriminator의 손실함수와 Generator의 손실함수를 출력한다. 
        if i % 100 == 0:
            print('(Epoch): %d, Generator loss(g_loss): %f, Discriminator loss(d_loss): %f' % (i, g_loss_print, d_loss_print))
            
            

