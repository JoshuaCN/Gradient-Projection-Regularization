import tensorflow as tf


def generate_pgd_common(x,
                        bounds,
                        model_fn,
                        attack_params,
                        one_hot_labels,
                        perturbation_multiplier,
                        random_init):
    """Common code for generating PGD adversarial examples.

    Args:
      x: original examples.
      bounds: tuple with bounds of image values, bounds[0] < bounds[1].
      model_fn: model function with signature model_fn(images).
      attack_params: parameters of the attack.
      one_hot_labels: one hot label vector to use in the loss.
      perturbation_multiplier: multiplier of adversarial perturbation,
        either +1.0 or -1.0.

    Returns:
      Tensor with adversarial examples.

    Raises:
      ValueError: if attack parameters are invalid.
    """
    # parse attack_params
    # Format of attack_params: 'EPS_STEP_NITER'
    # where EPS - epsilon, STEP - step size, NITER - number of iterations
    params_list = attack_params.split('_')
    if len(params_list) != 3:
        raise ValueError('Invalid parameters of PGD attack: %s' % attack_params)
    epsilon = float(params_list[0])
    step_size = float(params_list[1])
    niter = int(params_list[2])

    # rescale epsilon and step size to image bounds
    epsilon = epsilon / 255.0 * (bounds[1] - bounds[0])
    step_size = step_size / 255.0 * (bounds[1] - bounds[0])

    # clipping boundaries
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    # compute starting point
    if random_init:
        start_x = x + tf.keras.backend.random_uniform(tf.shape(x), -epsilon, epsilon)
        start_x = tf.clip_by_value(start_x, clip_min, clip_max)
        adv_images = tf.identity(start_x)
    else:
        adv_images = tf.identity(x)

    for i in range(niter):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            logits = model_fn(adv_images)
            loss = tf.nn.softmax_cross_entropy_with_logits(one_hot_labels, logits)
        perturbation = step_size * tf.sign(tape.gradient(loss, adv_images))
        adv_images += perturbation_multiplier * perturbation
        adv_images = tf.clip_by_value(adv_images, clip_min, clip_max)
    return adv_images


def generate_pgd(x, bounds, model_fn, attack_params, random_init):
    # pylint: disable=g-doc-args
    """Generats non-targeted PGD adversarial examples.

    See generate_pgd_common for description of arguments.

    Returns:
      tensor with adversarial examples.
    """
    # pylint: enable=g-doc-args

    # compute one hot predicted class
    logits = model_fn(x)
    num_classes = tf.shape(logits)[1]
    one_hot_labels = tf.one_hot(tf.argmax(model_fn(x), axis=1), num_classes)

    return generate_pgd_common(x, bounds, model_fn, attack_params,
                               one_hot_labels=one_hot_labels,
                               perturbation_multiplier=1.0, random_init=random_init)


def generate_adversarial_examples(x, bounds, model_fn, attack_description, random_init):
    """Generates adversarial examples.

    Args:
      x: original examples.
      bounds: tuple with bounds of image values, bounds[0] < bounds[1]
      model_fn: model function with signature model_fn(images).
      attack_description: string which describes an attack, see notes below for
        details.

    Returns:
      Tensor with adversarial examples.

    Raises:
      ValueError: if attack description is invalid.


    Attack description could be one of the following strings:
    - "clean" - no attack, return original images.
    - "pgd_EPS_STEP_NITER" - non-targeted PGD attack.
    - "pgdll_EPS_STEP_NITER" - tageted PGD attack with least likely target class.
    - "pgdrnd_EPS_STEP_NITER" - targetd PGD attack with random target class.

    Meaning of attack parameters is following:
    - EPS - maximum size of adversarial perturbation, between 0 and 255.
    - STEP - step size of one iteration of PGD, between 0 and 255.
    - NITER - number of iterations.
    """
    if attack_description == 'clean':
        return x
    idx = attack_description.find('_')
    if idx < 0:
        raise ValueError('Invalid value of attack description %s'
                         % attack_description)
    attack_name = attack_description[:idx]
    attack_params = attack_description[idx + 1:]
    if attack_name == 'pgd':
        return generate_pgd(x, bounds, model_fn, attack_params, random_init)
    else:
        raise ValueError('Invalid value of attack description %s'
                         % attack_description)
