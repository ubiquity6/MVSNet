from cluster_generator import ClusterGenerator
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import argparse
import time


def generator(n, mode='training'):
    train_gen = ClusterGenerator(args.data_dir, 5, 640, 480,
                                 128, 1, 8, mode=mode, flip_cams=False)
    return iter(train_gen)


def dataset(n):
    generator_data_type = (tf.float32, tf.float32, tf.float32)
    training_set = tf.data.Dataset.from_generator(
        lambda: generator(n, mode='validation'), generator_data_type)
    training_set = training_set.batch(1)
    training_set = training_set.prefetch(buffer_size=1)

    return training_set


def main(args):
    training_sample_size = 1000
    training_set = tf.data.Dataset.range(args.num_generators).apply(tf.data.experimental.parallel_interleave(
        dataset, cycle_length=args.num_generators, prefetch_input_elements=args.num_generators))
    training_iterator = training_set.make_initializable_iterator()

    images, cams, depth_image = training_iterator.get_next()

    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.inter_op_parallelism_threads = 0
    config.intra_op_parallelism_threads = 0
    start = time.time()

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run(training_iterator.initializer)
        for i in range(training_sample_size):
            im, cam, depth = sess.run([images, cams, depth_image])
            if i % 40 == 0:
                print(
                    '\n --- Time at cluster {} is {} --- \n'.format(i, time.time()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/Users/chrisheinrich/data/7scene-data/test')
    parser.add_argument('--num_generators', type=int,
                        default=4)
    args = parser.parse_args()
    main(args)
