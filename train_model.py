from tensorflow.keras.layers import Conv2D, Dense, Input, Activation, Dropout, Flatten, BatchNormalization, \
    GlobalAveragePooling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def modelA():
    image = Input(shape=input_shape)
    y = Conv2D(filters=64, kernel_size=5, padding='valid', activation='relu')(image)
    y = Conv2D(filters=64, kernel_size=5, activation='relu')(y)
    y = Dropout(0.25)(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.5)(y)
    logits = Dense(10, activation='linear')(y)
    model = Model(image, logits)
    return model


def modelB():
    image = Input(shape=input_shape)
    y = Dropout(0.2)(image)
    y = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same', activation='relu')(y)
    y = Conv2D(filters=128, kernel_size=6, strides=(2, 2), padding='valid', activation='relu')(y)
    y = Conv2D(filters=128, kernel_size=5, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Flatten()(y)
    logits = Dense(10, activation='linear')(y)
    model = Model(image, logits)
    return model


def modelC():
    image = Input(shape=input_shape)
    y = Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu')(image)
    y = Conv2D(filters=64, kernel_size=3, activation='relu')(y)
    y = Dropout(0.25)(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.5)(y)
    logits = Dense(10, activation='linear')(y)
    model = Model(image, logits)
    return model


def modelD():
    image = Input(shape=input_shape)
    y = Flatten()(image)
    y = Dense(300, kernel_initializer='he_normal', activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(300, kernel_initializer='he_normal', activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(300, kernel_initializer='he_normal', activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(300, kernel_initializer='he_normal', activation='relu')(y)
    y = Dropout(0.5)(y)
    logits = Dense(10, activation='linear')(y)
    model = Model(image, logits)
    return model


def model_mnist(type=0):
    """
    Defines MNIST model using Keras sequential model
    """

    models = [modelA, modelB, modelC, modelD]

    return models[type]()


def residual_network(img_input, classes_num=10, stack_n=5):
    weight_decay = 1e-4

    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='linear', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = tf.keras.layers.Conv2D(num_filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
        x = conv(x)
    return x


def l2_loss(model):
    weight_decay = 1e-4
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weight_decay


def model_cifar(type=0):
    num_classes = 10
    img_rows, img_cols = 32, 32
    img_channels = 3
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    resnet8 = Model(img_input, residual_network(img_input, num_classes, 1))
    resnet20 = Model(img_input, residual_network(img_input, num_classes, 3))
    models = [resnet8, resnet20]
    return models[type]


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from training_method import *

    dataset = 'mnist'
    AUTO = tf.data.experimental.AUTOTUNE

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train[..., None] / np.float32(255), x_test[..., None] / np.float32(255)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        num_epochs = 12
        batch_size = 256
        eps, sigma = 76 / 255, 76 / 255
        optimizer = tf.keras.optimizers.Adam()

    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        num_epochs = 50
        batch_size = 64
        eps, sigma = 8 / 255, 8 / 255
        iterations = x_train.shape[0] // batch_size + 1
        learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([40 * iterations, 60 * iterations],
                                                                             [1e-1, 1e-2, 1e-3])
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)

    input_shape = x_train[0].shape
    bounds = (0, 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


    def data_augment(image, label):
        # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
        # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
        # of the TPU while the TPU itself is computing gradients.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
        image = tf.image.random_crop(image, [32, 32, 3])

        return image, label


    def get_training_dataset():
        data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # data = data.repeat() # Since we use custom training loop, we don't need to use repeat() here.
        # data = data.shuffle(2048)
        if dataset == 'cifar10':
            data = data.map(data_augment, num_parallel_calls=AUTO)
        data = data.batch(batch_size)
        data = data.prefetch(AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
        return data


    loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    reg_loss = tf.keras.metrics.Mean()
    train_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)

    from tensorflow.keras.utils import Progbar

    model = model_mnist(0)  # ！！！
    source = []
    for i in range(4):  # ！！！
        source.append(tf.keras.models.load_model('./Models/ens/model_' + str(i) + '.h5'))

    for epoch in range(num_epochs):
        train_dataset = get_training_dataset()
        print("\nepoch {}/{}".format(epoch + 1, num_epochs))
        pb_i = Progbar(x_train.shape[0], stateful_metrics=['loss', 'acc'], verbose=2)
        for x, y in train_dataset:
            i = np.random.choice([0, 2, 3])  # ！！！

            # PGD Adversarial Training
            # adv_train(model,x,y,adv_params='pgd_76_10_8',random_init=True) #！！！

            # Ensemble Adversarial Training
            ## x_adv = generate_adversarial_examples(x, bounds, source[i], 'pgd_76_76_1', random_init=False)
            ## normal_train(model,x,y)
            # ens_train(source[i],model,x,y,adv_params='pgd_76_76_1',random_init=False) #！！！ 

            # Projection Regularization
            pr_train(source[i], model, x, y, alpha=0.3, beta=10, sigma=sigma)  # ！！！ 

            # Combination 1
            # x_adv = generate_adversarial_examples(x, bounds, source[i], 'pgd_76_76_1', random_init=True)
            # pr_train(source[i], model, x_adv, y, alpha=0.3, beta=10, sigma=sigma) #！！！

            # Combination 2
            # ens_pr_train(source[i], model, x, x_adv, y, alpha=0.3, beta=10, sigma=sigma) #！！！

            values = [('loss', train_loss.result()), ('acc', train_accuracy.result())]
            pb_i.add(x.shape[0], values=values)
        if (epoch + 1) % 3 == 0:
            for x, y in test_dataset:
                test_accuracy(y, model(x, training=False))
                test_loss(y, model(x, training=False))
            print('test loss:%.4f test acc:%.4f' % (test_loss.result().numpy(), test_accuracy.result().numpy()))
            test_loss.reset_states()
            test_accuracy.reset_states()
        # Reset training metrics at the end of each epoch
        train_accuracy.reset_states()
        train_loss.reset_states()
    model.save('./Models/ens/pr_model_' + str(0) + '.h5')
    del model

