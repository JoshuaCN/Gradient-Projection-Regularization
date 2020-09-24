def grid_visual(data):
    """
      This function displays a grid of images to show full misclassification
      :param data: grid data of the form;
          [nb_classes : nb_classes : img_rows : img_cols : nb_channels]
      :return: if necessary, the matplot figure to reuse
    """
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = [15, 15]
    # Ensure interactive mode is disabled and initialize our graph
    plt.ioff()
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')

    # Add the images to the plot
    num_cols = data.shape[0]
    num_rows = data.shape[1]
    num_channels = data.shape[4]
    for y in range(num_rows):
        for x in range(num_cols):
            figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
            plt.axis('off')

            if num_channels == 1:
                plt.imshow(data[x, y, :, :, 0], cmap='gray')
            else:
                plt.imshow(data[x, y, :, :, :])

    # Draw the plot and return
    plt.show()
    return figure


if __name__ == '__main__':
    from art.attacks.evasion import FastGradientMethod as FGM, ProjectedGradientDescentTensorFlowV2 as PGD, \
        CarliniL2Method as CW
    from art.classifiers import TensorFlowV2Classifier as Wrapper
    from art.utils import compute_accuracy, compute_success
    import tensorflow as tf
    import numpy as np

    dataset = 'mnist'
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
    input_shape = x_train[0].shape
    m = 1000
    eps = 0.3
    attacks = [
        'FGM',
        'PGD',
        # 'CW',
    ]

    FGM_params = {
        'eps': eps,
        'norm': np.inf,
        'batch_size': 500,
        'num_random_init': 0
    }

    PGD_params = {
        'eps': eps,
        'eps_step': 0.02,  # ！！！
        'max_iter': 30,
        'norm': np.inf,
        'batch_size': 500,
        'num_random_init': 0
    }

    CW_params = {
        'learning_rate': .1,
        'confidence': 1,
        'batch_size': 1000,
        'max_iter': 100,
    }
    loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
    source_models = []
    source_models.append(tf.keras.models.load_model('./Models/ens/model_1.h5'))
    target_models = []
    # target_models.append(tf.keras.models.load_model('./Models/ens/ens_model_0.h5')) # 集成对抗训练
    target_models.append(tf.keras.models.load_model('./Models/ens/pr_model_0.h5'))  # 投影正则化
    # target_models.append(tf.keras.models.load_model('./Models/ens/model_0.h5')) # 无防御
    # target_models.append(tf.keras.models.load_model('./Models/ens/pgd_model_0.h5')) # PGD对抗训练

    art = Wrapper(source_models[0], nb_classes=10, clip_values=(0, 1), input_shape=input_shape, loss_object=loss_fn)
    for attack in attacks:
        print('Performing' + attack + 'attack:')
        adv_acc = []
        for eps in [0, 0.2, 0.3, 0.4]:  # ！！！
            if eps > 0:
                FGM_params['eps'] = eps
                PGD_params['eps'] = eps
                adversary = eval(attack + '(art, **' + attack + '_params)')
                x_adv = adversary.generate(x_test[:m])
            else:
                x_adv = x_test[:m]
            for target in target_models:
                adv_acc.append(compute_accuracy(target.predict(x_adv), y_test[:m])[0])
                # print('Target Accuracy=', round(target_acc*100, 2),'%')
        # row = (lambda a: 1 if a < 10 else (10 if a > 100 else a//10))(m)
        # column = np.amin([10, m])
        # grid_visual(x_adv[:int(row * column)].reshape(column, row, input_shape[0], input_shape[1], input_shape[2]))
        print(adv_acc)
