# This is the preprocess module for autonomous learning. This module will take a dataset as input (labeled and
# unlabeled, or only labeled, should be clarified in the arguments) and automatically generate labeled and unlabeled
# datasets for the ssl module

import tensorflow as tf

import numpy as np
import os


def read_data(path_to_data, HEIGHT, WIDTH, DEPTH):
    # read images from datapath
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, DEPTH, HEIGHT, WIDTH))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_labels(path_to_labels):
    # read labels from datapath
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def _list_to_tf_dataset(dataset):
    # transform list to tf dataset
    def _dataset_gen():
        for example in dataset:
            example['image']=tf.image.resize(example['image'],(32,32))
            yield example
    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={'image': tf.uint8, 'label': tf.int64},
        output_shapes={'image': (32, 32, 3), 'label': ()}
    )


def load_unlabeled(UNLAB_PATH, HEIGHT, WIDTH, DEPTH, num_unlabels):
    # load unlabeled examples to list, and later transformed to tf dataset
    train = []
    images = read_data(UNLAB_PATH, HEIGHT, WIDTH, DEPTH)
    for index, image in enumerate(images):
        if index>=num_unlabels:
            break
        train.append({
            "image": image,
            "label": -1,
        })
    train = _list_to_tf_dataset(train)
    return train


def load_labeled(DATA_PATH, LABEL_PATH, TEST_PATH, TEST_LABEL_PATH, HEIGHT , WIDTH, DEPTH):
    # load labeled examples to list, and later transformed to tf dataset
    train = []
    images = read_data(DATA_PATH, HEIGHT , WIDTH, DEPTH)
    labels = read_labels(LABEL_PATH)
    for index, image in enumerate(images):
        train.append({
            "image": image,
            "label": labels[index] - 1 if labels is not None else -1,
        })

    test = []
    test_images = read_data(TEST_PATH, HEIGHT, WIDTH, DEPTH)
    test_labels = read_labels(TEST_LABEL_PATH)
    for index, image in enumerate(test_images):
        test.append({
            "image": image,
            "label": test_labels[index] - 1 if test_labels is not None else -1,
        })
    train = _list_to_tf_dataset(train)
    test = _list_to_tf_dataset(test)
    return train, test


def split_data(dataset, num_classes, num_validations):
    # train/validation split for ssl input
    dataset = dataset.shuffle(buffer_size=10000)
    counter = [0 for _ in range(num_classes)]
    labelled = []
    validation = []
    for example in iter(dataset):
        label = int(example['label'])
        counter[label] += 1
        if counter[label] <= (num_validations / num_classes):
            validation.append(example)
            continue
        else:
            labelled.append(example)
    labelled = _list_to_tf_dataset(labelled)
    validation = _list_to_tf_dataset(validation)
    return labelled, validation


def split_labeled(dataset, num_labelled, num_unlabeled, num_validations, num_classes):
    # delabel and train/validation split for sl input
    dataset = dataset.shuffle(buffer_size=10000)
    counter = [0 for _ in range(num_classes)]
    count_unlabeled=0
    labelled = []
    unlabelled = []
    validation = []
    for example in iter(dataset):
        label = int(example['label'])
        counter[label] += 1
        if counter[label] <= (num_labelled / num_classes):
            labelled.append(example)
            continue
        elif counter[label] <= (num_validations / num_classes + num_labelled / num_classes):
            validation.append(example)
        elif count_unlabeled <= num_unlabeled:
            unlabelled.append({
                'image': example['image'],
                'label': tf.convert_to_tensor(-1, dtype=tf.int64)
            })
            count_unlabeled+=1
    labelled = _list_to_tf_dataset(labelled)
    unlabelled = _list_to_tf_dataset(unlabelled)
    validation = _list_to_tf_dataset(validation)
    return labelled, unlabelled, validation


def _bytes_feature(value):
    # bytes feature for serialization
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    # int feature for serialization
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, label):
    # serialization of dataset
    image = tf.image.encode_png(image)
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def tf_serialize_example(example):
    # serialization for export
    tf_string = tf.py_function(
        serialize_example,
        (example['image'], example['label']),
        tf.string
    )
    return tf.reshape(tf_string, ())


def export_tfrecord_dataset(dataset_path, dataset):
    # store the processed data for future use
    serialized_dataset = dataset.map(tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(dataset_path)
    writer.write(serialized_dataset)


def _parse_function(example):
    # parse description
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)


def load_tfrecord_dataset(dataset_path):
    # load the stored processed data
    raw_dataset = tf.data.TFRecordDataset([dataset_path])
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def normalize_image(image, start=(0., 255.), end=(-1., 1.)):
    # image normalization
    image = (image - start[0]) / (start[1] - start[0])
    image = image * (end[1] - end[0]) + end[0]
    return image


def process_parsed_dataset(dataset, num_classes):
    # process the parsed dataset
    images = []
    labels = []
    for example in iter(dataset):
        decoded_image = tf.io.decode_png(example['image'], channels=3, dtype=tf.uint8)
        normalized_image = normalize_image(tf.cast(decoded_image, dtype=tf.float32))
        images.append(normalized_image)
        one_hot_label = tf.one_hot(example['label'], depth=num_classes, dtype=tf.float32)
        labels.append(one_hot_label)
    return tf.data.Dataset.from_tensor_slices({
        'image': images,
        'label': labels
    })


def preprocess_dataset(args, log_dir):
    # preprocess module
    dataset_path = f'{log_dir}/datasets'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    num_classes = args['num_classes']

    DATA_PATH = args['path_train']
    LABEL_PATH = args['path_label']
    TEST_PATH = args['path_test']
    TEST_LABEL_PATH = args['path_testlabel']
    UNLAB_PATH = args['path_unlabeled']

    HEIGHT = args['image_height']
    WIDTH = args['image_width']
    DEPTH = args['image_depth']

    # creating datasets
    if args['delabel_train']==False:
        train, test = load_labeled(DATA_PATH, LABEL_PATH, TEST_PATH, TEST_LABEL_PATH, HEIGHT, WIDTH, DEPTH)
        trainU = load_unlabeled(UNLAB_PATH, HEIGHT, WIDTH, DEPTH, args['unlabelled_examples'])
        trainX, validation = split_data(train, num_classes, args['validation_examples'])
    else:
        train, test = load_labeled(DATA_PATH, LABEL_PATH, TEST_PATH, TEST_LABEL_PATH, HEIGHT, WIDTH, DEPTH)
        trainX, trainU, validation = split_labeled(train, args['labelled_examples'], args['unlabelled_examples'],
                                                    args['validation_examples'], num_classes)


    for name, dataset in [('trainX', trainX), ('trainU', trainU), ('validation', validation), ('test', test)]:
        export_tfrecord_dataset(f'{dataset_path}/{name}.tfrecord', dataset)

    # saving datasets as .tfrecord files
    export_tfrecord_dataset(f'{dataset_path}/trainX.tfrecord', trainX)
    export_tfrecord_dataset(f'{dataset_path}/trainU.tfrecord', trainU)
    export_tfrecord_dataset(f'{dataset_path}/validation.tfrecord', validation)
    export_tfrecord_dataset(f'{dataset_path}/test.tfrecord', test)

    # loading datasets from .tfrecord files
    parsed_trainX = load_tfrecord_dataset(f'{dataset_path}/trainX.tfrecord')
    parsed_trainU = load_tfrecord_dataset(f'{dataset_path}/trainU.tfrecord')
    parsed_validation = load_tfrecord_dataset(f'{dataset_path}/validation.tfrecord')
    parsed_test = load_tfrecord_dataset(f'{dataset_path}/test.tfrecord')

    trainX = process_parsed_dataset(parsed_trainX, num_classes)
    trainU = process_parsed_dataset(parsed_trainU, num_classes)
    validation = process_parsed_dataset(parsed_validation, num_classes)
    test = process_parsed_dataset(parsed_test, num_classes)

    return trainX, trainU, validation, test, num_classes
