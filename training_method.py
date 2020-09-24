import tensorflow as tf
from pgd import generate_adversarial_examples


def square_cosine_similarity(a, b, axis=-1):
    a = tf.nn.l2_normalize(a, axis=axis)
    b = tf.nn.l2_normalize(b, axis=axis)
    return tf.square(tf.reduce_sum(a * b, axis=axis))


def square_projection(a, b, axis=-1):
    # square projection of target graident(b) to source gradient(a)
    a = tf.nn.l2_normalize(a, axis=axis)
    return tf.square(tf.reduce_sum(a * b, axis=axis))


@tf.function
def normal_train(model, x, y):
    """
    normal training
    """
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        loss = loss_fn(y, outputs)
        train_accuracy(y, outputs)
        train_loss(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


@tf.function
def ens_train(source, model, x, y, **kwargs):
    with tf.GradientTape() as tape:
        tape.watch(x)
        outputs = model(x, training=True)
        x_adv = generate_adversarial_examples(x, bounds, source, kwargs['adv_params'], kwargs['random_init'])
        loss_g = loss_fn(y, outputs)
        loss_adv = loss_fn(y, model(x_adv, training=True))
        train_accuracy(y, outputs)
        train_loss(loss_g)
        reg_loss(loss_adv)
        total_loss = 0.5 * loss_g + 0.5 * loss_adv
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


@tf.function
def adv_train(model, x, y, **kwargs):
    """
    adversarial training
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        outputs = model(x, training=True)
        x_adv = generate_adversarial_examples(x, bounds, model, kwargs['adv_params'], kwargs['random_init'])
        loss_g = loss_fn(y, outputs)
        loss_adv = loss_fn(y, model(x_adv, training=True))
        train_accuracy(y, outputs)
        train_loss(loss_g)
        reg_loss(loss_adv)
        total_loss = 0.5 * loss_g + 0.5 * loss_adv
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# @tf.function
# def spr_train(model_g, x, y, **kwargs):
#     grads_f = tf.TensorArray(x.dtype, size=kwargs['m'])
#     grads_g = tf.TensorArray(x.dtype, size=kwargs['m'])
#     with tf.GradientTape(persistent=True) as tape:
#         with tf.GradientTape() as tape2:
#             tape.watch(x)
#             outputs_g = model_g(x, training=True)
#             for i in tf.range(kwargs['m']):
#                 grads_g = grads_g.write(i, tape.gradient(loss_fn(
#                     y, model_g(x + tf.random.normal(x.shape, 0, kwargs['sigma']))), x))
#                 grads_f = grads_f.write(i, tape.gradient(loss_fn(
#                     y, kwargs['f'](x + tf.random.normal(x.shape, 0, kwargs['sigma']))), x))
#             Grads_f = tf.reshape(tf.reduce_mean(grads_f.stack(), axis=0), [x.shape[0], -1])
#             Grads_g = tf.reshape(tf.reduce_mean(grads_g.stack(), axis=0), [x.shape[0], -1])
#             sim_loss = tf.reduce_sum(square_cosine_similarity(Grads_f, Grads_g))
#             basic_loss = loss_fn(y, outputs_g)
#             total_loss = basic_loss + kwargs['alpha'] * sim_loss + kwargs['beta'] * tf.nn.l2_loss(Grads_g)
#             grads = tape2.gradient(total_loss, model_g.trainable_variables)
#             reg_loss(total_loss - basic_loss)
#             train_accuracy(y, outputs_g)
#             train_loss(basic_loss)
#     optimizer.apply_gradients(zip(grads, model_g.trainable_variables))


@tf.function
def pr_train(source, model, x, y, **kwargs):
    with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape() as tape2:
            tape.watch(x)
            outputs_g = model(x + tf.random.normal(x.shape, 0, kwargs['sigma']), training=True)
            cls_loss_g = loss_fn(y, outputs_g)
            grads_g = tape.gradient(cls_loss_g, x)
            outputs_f = source(x + tf.random.normal(x.shape, 0, kwargs['sigma']), training=False)
            cls_loss_f = loss_fn(y, outputs_f)
            grads_f = tape.gradient(cls_loss_f, x)
            sim_loss = tf.reduce_sum(square_cosine_similarity(tf.reshape(grads_f, [x.shape[0], -1]),
                                                              tf.reshape(grads_g, [x.shape[0], -1])))
            total_loss = cls_loss_g + kwargs['alpha'] * sim_loss + kwargs['beta'] * tf.nn.l2_loss(grads_g)
            grads = tape2.gradient(total_loss, model.trainable_variables)
            train_accuracy(y, outputs_g)
            train_loss(cls_loss_g)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


@tf.function
def ens_pr_train(source, model, x, x_adv, y, **kwargs):
    with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape() as tape2:
            tape.watch(x)
            outputs_g = model(x + tf.random.normal(x.shape, 0, kwargs['sigma']), training=True)
            cls_loss_g = loss_fn(y, outputs_g)
            grads_g = tape.gradient(cls_loss_g, x)
            outputs_f = source(x + tf.random.normal(x.shape, 0, kwargs['sigma']), training=False)
            cls_loss_f = loss_fn(y, outputs_f)
            grads_f = tape.gradient(cls_loss_f, x)
            sim_loss = tf.reduce_sum(square_cosine_similarity(tf.reshape(grads_f, [x.shape[0], -1]),
                                                              tf.reshape(grads_g, [x.shape[0], -1])))
            total_loss = loss_fn(y, model(x_adv, training=True)) + kwargs['alpha'] * sim_loss + kwargs[
                'beta'] * tf.nn.l2_loss(grads_g)
            grads = tape2.gradient(total_loss, model.trainable_variables)
            train_accuracy(y, outputs_g)
            train_loss(cls_loss_g)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__ == '__main__':
    loss_fn = tf.losses.CategoricalCrossentropy()
    optimizer = tf.optimizers.Adam()
    train_accuracy = tf.metrics.CategoricalAccuracy()
    train_loss = tf.metrics.Mean()
    reg_loss = tf.metrics.Mean()
    bounds = (0, 1)

