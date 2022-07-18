# coding:UTF-8
from networks import Generator, Discriminator, generator_loss, discriminator_loss
import networks
import tensorflow as tf
import time
import datetime
import preprocess

# 实例化生成器和判别器对象
generator = Generator()
discriminator = Discriminator()
# 记录日志
# tensorboard --logdir=logs/fit
summary_writer = tf.summary.create_file_writer(
    "logs/" + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# 训练
def train_step(input_image, target, epoch):
    # 定义生成器梯度、判别器梯度
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 生成一张图片
        gen_output = generator(input_image, training=True)
        # 对真实图片进行判断
        disc_real_output = discriminator([input_image, target], training=True)
        # 对生成图片进行判断
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        # L1损失
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # 计算梯度
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
    # Adam优化
    networks.optimizerG.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    networks.optimizerD.apply_gradients(zip(discriminator_gradients,
                                            discriminator.trainable_variables))
    # 记录日志
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)

        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


EPOCHS = 100

if __name__ == '__main__':
    # 训练
    fit(preprocess.dataset, EPOCHS)
    # 保存模型
    generator.save('model/g_p_500.h5')
    discriminator.save('model/d_p_500.h5')
