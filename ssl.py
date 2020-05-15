# This is the ssl module for autonomous learning
# The labeled and unlabeled examples will go through random augmentation, label guessing, temperature sharpening, and
# mixup method. This module will return a training set with images and their generated labels

import numpy as np
import tensorflow as tf


@tf.function
def augment(x, HEIGHT, WIDTH, DEPTH):
    # random horizontal flipping
    x = tf.image.random_flip_left_right(x)
    # random padding
    x = tf.pad(x, paddings=[(0, 0), (4, 4), (4, 4), (0, 0)], mode='REFLECT')
    # random crop
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(HEIGHT, WIDTH, DEPTH)), x)
    return x


def guess_labels(u_aug, model, K):
    # guess labels for unlabeled examples
    guessed = tf.nn.softmax(model(u_aug[0]), axis=1)
    for k in range(1, K):
        guessed = guessed + tf.nn.softmax(model(u_aug[k]), axis=1)
    guessed = guessed / K
    guessed = tf.stop_gradient(guessed)
    return guessed


@tf.function
def sharpening(p, T):
    # temperature sharpening
    return tf.pow(p, 1/T) / tf.reduce_sum(tf.pow(p, 1/T), axis=1, keepdims=True)


@tf.function
def mixup(x1, x2, y1, y2, beta):
    # mixup method
    beta = tf.maximum(beta, 1-beta)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


def interleave_offsets(batch, nu):
    # offsets in interleave
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    # interleave
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [tf.concat(v, axis=0) for v in xy]


def ssl(args, model, x, y, u, T, K, beta):
    # ssl module
    HEIGHT = args['image_height']
    WIDTH = args['image_width']
    DEPTH = args['image_depth']
    batch_size = x.shape[0]
    x_aug = augment(HEIGHT, WIDTH, DEPTH, x)
    u_aug = [None for _ in range(K)]
    for k in range(K):
        u_aug[k] = augment(HEIGHT, WIDTH, DEPTH, u)
    guessed = guess_labels(u_aug, model, K)
    qb = sharpening(guessed, tf.constant(T))
    U = tf.concat(u_aug, axis=0)
    qb = tf.concat([qb for _ in range(K)], axis=0)
    XU = tf.concat([x_aug, U], axis=0)
    XU_label = tf.concat([y, qb], axis=0)
    indices = tf.random.shuffle(tf.range(XU.shape[0]))
    W = tf.gather(XU, indices)
    W_label = tf.gather(XU_label, indices)
    XU, XU_label = mixup(XU, W, XU_label, W_label, beta=beta)
    XU = tf.split(XU, K + 1, axis=0)
    XU = interleave(XU, batch_size)
    return XU, XU_label


@tf.function
def ssl_loss(labels_x, logits_x, labels_u, logits_u):
    # use the combination of xe loss and l2 loss as the ssl loss
    loss_xe = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    loss_l2 = tf.square(labels_u - tf.nn.softmax(logits_u))
    loss_l2 = tf.reduce_mean(loss_l2)
    return loss_xe, loss_l2


def linear_rampup(epoch, rampup_length):
    # linear rampup for unlabeled weight
    if rampup_length == 0:
        return 1.
    else:
        rampup = np.clip(epoch / rampup_length, 0., 1.)
        return float(rampup)


def weight_decay(model, decay_rate):
    # weight decay during the training
    for var in model.trainable_variables:
        var.assign(var * (1 - decay_rate))


def ema(model, ema_model, ema_decay):
    # ema model for unlabeled weight
    for var, ema_var in zip(model.variables, ema_model.variables):
        if var.trainable:
            ema_var.assign((1 - ema_decay) * var + ema_decay * ema_var)
        else:
            ema_var.assign(tf.identity(var))
