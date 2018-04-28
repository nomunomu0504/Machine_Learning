# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

NUM_CLASSES = 9
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of test data')
flags.DEFINE_string('train_dir', 'tmp/data', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 30, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 256, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')


def inference(images_placeholder, keep_prob):
    """ Function to create a prediction model

        vals:
            image_placeholder: plocaholder od image.
            keep_prob: placeholder of parsent of dropout.

        return:
            y_conv: probability of class.
    """

    # initialize weight.
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # initialize Bias.
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Create Convolution layer.
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # Create Pooling layer.
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Transformation to 28x28x3.
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    # Create Convolution1.
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Create Pooling1 layer.
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    # Create Convolution2.
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Create Pooling2 layer.
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # Create Total bonding1 layer.
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout Setting.
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Create Total bonding2 layer.
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # Normalization by SoftMax function
    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Return something like the probability of each label
    return y_conv


def loss(logits, labels):
    """ Function to calculate loss
        
        Vals:
            logits: tensor pf logit, float - [batch_size, NUM_CLASSES]
            labels: tensor of labels, int32 - [batch_size, NUM_CLASSES]
            
        return:
            cross_entropy: float - tensor of cross-entropy.
    """

    # Calculate of cross-entropy.
    # cross_entropy = -tf.reduce_sum(labels * tf.log(logits))
    cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

    # Set View of TensorBoard.
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy


def training(loss, learning_rate):
    """ Function to define Op of training
    
        Vals:
            loss: tensor of loss, result of loss().
            learning_rate: practice rate.
            
        return:
            train_step: Op of train.
    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


def accuracy(logits, labels):
    """ Function to calculate of accuracy
    
        Vals:
            lagits: result of inference()
            labels: tensor of label, int32 - [batch_size, NUM_CLASSES]
            
        return:
            accuracy: float - accuracy
    """

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

if __name__ == '__main__':
    # Open file.
    f = open(FLAGS.train, 'r')

    # dimension of data.
    train_image = []
    train_label = []

    for line in f:
        try:
            # delete line-break, strip space.
            line = line.rstrip()
            l = line.split()

            # image resize to 96x96
            img = cv2.imread(l[0])
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            train_image.append(img.flatten().astype(np.float32) / 255.0)

            tmp = np.zeros(NUM_CLASSES)
            tmp[int(l[1])-1] = 1
            train_label.append(tmp)
        except Exception as e:
            print(e)

    # Convert to numpy type.
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    f.close()

    f = open(FLAGS.test, 'r')
    test_image = []
    test_label = []
    for line in f:
        # delete line-break, strip space.
        line = line.rstrip()
        l = line.split()

        # image resize to 28x28.
        img = cv2.imread(l[0])
        img = cv2.resize(img, (28, 28))

        test_image.append(img.flatten().astype(np.float32) / 255.0)

        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])-1] = 1
        test_label.append(tmp)

    # Convert to numpy type.
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()

    with tf.Graph().as_default():
        # Temporary Tensor for inserting images.
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))

        # Temporary Tensor for inserting labels.
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))

        # Temporary Tensor for inserting parsents of dropout.
        keep_prob = tf.placeholder("float")

        # Make models by Call inference().
        logits = inference(images_placeholder, keep_prob)

        # Call loss(), Calculate loss.
        loss_value = loss(logits, labels_placeholder)

        # Call training(), practice.
        train_op = training(loss_value, FLAGS.learning_rate)

        # Calculate accuracy.
        acc = accuracy(logits, labels_placeholder)

        # setting of save.
        saver = tf.train.Saver()

        # Create Session.
        session = tf.Session()

        # initialize of variables.
        session.run(tf.global_variables_initializer())

        # Set value, TensorBoard.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, session.graph)

        # exec training.
        for step in range(FLAGS.max_steps):
            for i in range(int(len(train_image) / FLAGS.batch_size)):

                # Execution of training for batch_size images
                batch = FLAGS.batch_size * i
                session.run(train_op, feed_dict={
                    images_placeholder: train_image[batch: batch + FLAGS.batch_size],
                    labels_placeholder: train_label[batch: batch + FLAGS.batch_size],
                    keep_prob: 0.5
                })

                print("i: %d" % i)

            train_accuracy = session.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0
            })

            print("step: %d, training accuracy %g" % (step, train_accuracy))

            summary_str = session.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0
            })
            summary_writer.add_summary(summary_str, step)

    print("test accuracy: %g" % session.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0
    }))

    save_path = saver.save(session, "model.ckpt")
