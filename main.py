import argparse
import os

import numpy as np
import tensorflow as tf
import tqdm
import yaml

from preprocess import preprocess_dataset
from ssl import ssl, ssl_loss, linear_rampup, interleave, weight_decay, ema
from model import WideResNet
from bayesian_optimization import Bayesian_Optimization

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
gpus = tf.config.experimental.list_physical_devices('GPU')
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.config.experimental.set_memory_growth(gpus[0], True)
session = tf.compat.v1.InteractiveSession(config=config)


def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=None, help='Random seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='stl10', help='Name of the chosen dataset')
    parser.add_argument('--num-classes', type=int, default=10, help='Name of the chosen dataset')

    parser.add_argument('--mode', type=str, default='training',
                        help='Whether to train the model for search for hyperparameters (training/tuning)')

    parser.add_argument('--delabel-train', type=bool, default=False,
                        help='Whether need to delabel the training set')
    parser.add_argument('--path-train', type=str, default='D:/stl10/train_X.bin', help='Path of labeled examples')
    parser.add_argument('--path-label', type=str, default='D:/stl10/train_y.bin', help='Path of labels')
    parser.add_argument('--path-unlabeled', type=str, default='D:/stl10/unlabeled_X.bin',
                        help='Path of unlabeled examples')
    parser.add_argument('--path-test', type=str, default='D:/stl10/test_X.bin', help='Path of test examples')
    parser.add_argument('--path-testlabel', type=str, default='D:/stl10/test_y.bin', help='Path of test labels')

    parser.add_argument('--image-height', type=int, default=96, help='Image height')
    parser.add_argument('--image-width', type=int, default=96, help='Image width')
    parser.add_argument('--image-depth', type=int, default=3, help='Image depth')

    parser.add_argument('--labelled-examples', type=int, default=1000, help='number of labelled examples')
    parser.add_argument('--unlabelled-examples', type=int, default=10000, help='number of unlabelled examples')
    parser.add_argument('--validation-examples', type=int, default=1000, help='number of validation examples')

    parser.add_argument('--epochs', type=int, default=128, help='Number of epochs')
    parser.add_argument('--batch-size',  type=int, default=64, help='examples per batch (default: 64)')

    parser.add_argument('--val-iteration', type=int, default=1024,
                        help='number of iterations before validation (default: 1024)')
    parser.add_argument('--T', type=float, default=0.5, help='temperature sharpening ratio (default: 0.5)')
    parser.add_argument('--K', type=int, default=2, help='number of rounds of augmentation (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='param for sampling from Beta distribution (default: 0.75)')
    parser.add_argument('--lambda-u', type=int, default=100, help='multiplier for unlabelled loss (default: 100)')
    parser.add_argument('--rampup-length', type=int, default=16,
                        help='rampup length for unlabelled loss multiplier (default: 16)')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='decay rate for model vars (default: 0.02)')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='ema decay for ema model vars (default: 0.999)')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='learning_rate, (default: 0.01)')

    parser.add_argument('--config-path', type=str, default=None, help='path to yaml config file, overwrites args')
    parser.add_argument('--tensorboard', action='store_true', help='enable tensorboard visualization')
    parser.add_argument('--resume', action='store_true', help='whether to restore from previous training runs')

    return parser.parse_args()


def load_config(args):
    # load config yaml file to set parameters
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, args['config_path'])
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args


def main():
    global datasetX, datasetU, val_dataset, model, ema_model, optimizer, epoch, args
    args = vars(get_args())
    epoch = args['epochs']
    start_epoch = 0
    record_path = f'.logs/{args["dataset"]}@{args["labelled_examples"]}'
    ckpt_dir = f'{record_path}/checkpoints'
    datasetX, datasetU, val_dataset, test_dataset, num_classes = preprocess_dataset(args, record_path)

    model = WideResNet(num_classes, depth=28, width=2)
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = tf.keras.optimizers.Adam(lr=args['learning_rate'])
    model_ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(model_ckpt, f'{ckpt_dir}/model', max_to_keep=3)

    ema_model = WideResNet(num_classes, depth=28, width=2)
    ema_model.build(input_shape=(None, 32, 32, 3))
    ema_model.set_weights(model.get_weights())
    ema_ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=ema_model)
    ema_manager = tf.train.CheckpointManager(ema_ckpt, f'{ckpt_dir}/ema', max_to_keep=3)

    if args['resume']:
        model_ckpt.restore(manager.latest_checkpoint)
        ema_ckpt.restore(manager.latest_checkpoint)
        model_ckpt.step.assign_add(1)
        ema_ckpt.step.assign_add(1)
        start_epoch = int(model_ckpt.step)
        print(f'Restored @ epoch {start_epoch} from {manager.latest_checkpoint} and {ema_manager.latest_checkpoint}')

    train_writer = None
    if args['tensorboard']:
        train_writer = tf.summary.create_file_writer(f'{record_path}/train')
        val_writer = tf.summary.create_file_writer(f'{record_path}/validation')
        test_writer = tf.summary.create_file_writer(f'{record_path}/test')

    args['T'] = tf.constant(args['T'])
    args['beta'] = tf.Variable(0., shape=())

    if args['mode']=='tuning':
        params=[datasetX, datasetU, val_dataset, model, ema_model, optimizer, epoch, args]
        Bayesian_Optimization(params)
    else:
        for epoch in range(start_epoch, args['epochs']):
            xe_loss, l2u_loss, total_loss, accuracy = train(datasetX, datasetU, model, ema_model, optimizer, epoch,
                                                            args)
            val_xe_loss, val_accuracy = validate(val_dataset, ema_model, epoch, args, split='Validation')
            test_xe_loss, test_accuracy = validate(test_dataset, ema_model, epoch, args, split='Test')

            if (epoch - start_epoch) % 16 == 0:
                model_save_path = manager.save(checkpoint_number=int(model_ckpt.step))
                ema_save_path = ema_manager.save(checkpoint_number=int(ema_ckpt.step))
                print(f'Saved model checkpoint for epoch {int(model_ckpt.step)} @ {model_save_path}')
                print(f'Saved ema checkpoint for epoch {int(ema_ckpt.step)} @ {ema_save_path}')

            model_ckpt.step.assign_add(1)
            ema_ckpt.step.assign_add(1)

            step = args['val_iteration'] * (epoch + 1)
            if args['tensorboard']:
                with train_writer.as_default():
                    tf.summary.scalar('xe_loss', xe_loss.result(), step=step)
                    tf.summary.scalar('l2u_loss', l2u_loss.result(), step=step)
                    tf.summary.scalar('total_loss', total_loss.result(), step=step)
                    tf.summary.scalar('accuracy', accuracy.result(), step=step)
                with val_writer.as_default():
                    tf.summary.scalar('xe_loss', val_xe_loss.result(), step=step)
                    tf.summary.scalar('accuracy', val_accuracy.result(), step=step)
                with test_writer.as_default():
                    tf.summary.scalar('xe_loss', test_xe_loss.result(), step=step)
                    tf.summary.scalar('accuracy', test_accuracy.result(), step=step)

    if args['tensorboard']:
        for writer in [train_writer, val_writer, test_writer]:
            writer.flush()


def train(datasetX, datasetU, model, ema_model, optimizer, epoch, args):
    xe_loss_avg = tf.keras.metrics.Mean()
    l2u_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)

    iteratorX = iter(shuffle_and_batch(datasetX))
    iteratorU = iter(shuffle_and_batch(datasetU))

    progress_bar = tqdm.tqdm(range(args['val_iteration']), unit='batch')
    for batch_num in progress_bar:
        lambda_u = args['lambda_u'] * linear_rampup(epoch + batch_num/args['val_iteration'], args['rampup_length'])
        try:
            batchX = next(iteratorX)
        except:
            iteratorX = iter(shuffle_and_batch(datasetX))
            batchX = next(iteratorX)
        try:
            batchU = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(datasetU))
            batchU = next(iteratorU)

        args['beta'].assign(np.random.beta(args['alpha'], args['alpha']))
        with tf.GradientTape() as tape:
            XU, XUy = ssl(args, model, batchX['image'], batchX['label'], batchU['image'], args['T'], args['K'], args['beta'])
            logits = [model(XU[0])]
            for batch in XU[1:]:
                logits.append(model(batch))
            logits = interleave(logits, args['batch_size'])
            logits_x = logits[0]
            logits_u = tf.concat(logits[1:], axis=0)

            xe_loss, l2u_loss = ssl_loss(XUy[:args['batch_size']], logits_x, XUy[args['batch_size']:], logits_u)
            total_loss = xe_loss + lambda_u * l2u_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema(model, ema_model, args['ema_decay'])
        weight_decay(model=model, decay_rate=args['weight_decay'] * args['learning_rate'])

        xe_loss_avg(xe_loss)
        l2u_loss_avg(l2u_loss)
        total_loss_avg(total_loss)
        accuracy(tf.argmax(batchX['label'], axis=1, output_type=tf.int32), model(tf.cast(batchX['image'], dtype=tf.float32), training=False))

        progress_bar.set_postfix({
            'XE Loss': f'{xe_loss_avg.result():.4f}',
            'L2U Loss': f'{l2u_loss_avg.result():.4f}',
            'WeightU': f'{lambda_u:.3f}',
            'Total Loss': f'{total_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
    return xe_loss_avg, l2u_loss_avg, total_loss_avg, accuracy


def validate(dataset, model, epoch, args, split):
    accuracy = tf.keras.metrics.Accuracy()
    xe_avg = tf.keras.metrics.Mean()

    dataset = dataset.batch(args['batch_size'])
    for batch in dataset:
        logits = model(batch['image'], training=False)
        xe_loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch['label'], logits=logits)
        xe_avg(xe_loss)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy(prediction, tf.argmax(batch['label'], axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} XE Loss: {xe_avg.result():.4f}, {split} Accuracy: {accuracy.result():.3%}')

    return xe_avg, accuracy


if __name__ == '__main__':
    main()
