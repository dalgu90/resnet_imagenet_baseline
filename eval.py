import numpy as np
from IPython import embed
import tensorflow as tf

import model
#import awa_input as data_input
import imagenet_input as data_input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', './train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('output', 'output.txt',
                           """Output text file.""")
tf.app.flags.DEFINE_boolean('train_data', False,
                           """To use training data in evaluation.""")
tf.app.flags.DEFINE_boolean('embed', False,
                           """Open IPython shell after building model and restoring from checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95,
                            """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_string('train_dir', './train_loss_weight2',
                           """Directory where to load pretrained model.absolute_importgj""")

DATASET_CLASS_PATH = data_input.CLASS_LIST_FPATH
#DATASET_ATTRIBUTE_PATH = '/data/common_datasets/AwA/Animals_with_Attributes/predicates.txt'
#DATASET_ATTRIBUTE_PATH = '/home/dalgu/dataset/apascal/attribute_names_apy25.txt'

def evaluate():
    print('[Testing Configuration]')
    print('\tCheckpoint Dir: %s' % FLAGS.checkpoint_dir)
    print('\tDataset: %s data' % ('Training' if FLAGS.train_data else 'Test'))
    print('\tOutput: %s' % FLAGS.output)

    with open(DATASET_CLASS_PATH, 'r') as fd:
#        attribute_list = [temp.strip() for temp in fd.readlines()]
      class_list = [temp.strip()[10:40] for temp in fd.readlines()]
    batch_size = FLAGS.batch_size
    # num_attr = model.NUM_ATTRS
    num_class = model.NUM_CLASSES

    with tf.Graph().as_default():
        test_images, test_labels = model.inputs('train' if FLAGS.train_data else 'eval', False)
        test_logits = model.inference(test_images)

        if FLAGS.train_data:
            total_cnt = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            total_cnt = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


        # Start running operations on the Graph.
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=False))
        sess.run(init)

        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print '\tRestore from %s' % ckpt.model_checkpoint_path
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Open IPython shell, if flag set.
        if FLAGS.embed:
            embed()

        # Start the queue runners.
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        # correct, incorrect
        result_ll = [[0, 0] for _ in range(num_class)]
        idx = 0
        while idx < total_cnt:
            print('Current Idx: '  + str(idx))
            end = min([idx + batch_size, total_cnt])
            this_batch_size = end - idx

            logits_val, labels_val = sess.run([test_logits, test_labels])
            preds = np.argmax(logits_val, axis=1)
            for i in range(this_batch_size):
                correct = 0 if labels_val[i] == preds[i] else 1
                result_ll[labels_val[i] % num_class][correct] += 1
            idx = end

        coord.request_stop()
        coord.wait_for_stop(1)
        sess.close()


    accuracy = [float(result[0])/float(result[0]+result[1]) for result in result_ll]
    correct_t = np.sum([result[0] for result in result_ll])
    incorrect_t = np.sum([result[1] for result in result_ll])
    accuracy_t = float(correct_t)/float(correct_t + incorrect_t)

    print 'Class    \t\t\tT\tF\tAcc.'
    format_str = '%-31s %7d %7d %.5f'
    for i in range(num_class):
        print format_str % (class_list[i], result_ll[i][0], result_ll[i][1], accuracy[i])
    print(format_str % ('(Total)', correct_t, incorrect_t, accuracy_t))
    with open(FLAGS.output, 'w') as fd:
        fd.write('Class    \t\t\tT\tF\tAcc.\n')
        for i in range(num_class):
            t, f = result_ll[i]
            format_str = '%-31s %7d %7d %.5f\n'
            fd.write(format_str % (class_list[i].replace(' ', '-'),t,f,accuracy[i]))
        fd.write(format_str % ('(Total)',correct_t,incorrect_t,accuracy_t))

#     with open(FLAGS.output, 'w') as fd:
#         fd.write('Class    \tT\tF\tAcc.\n')
#         format_str = '%-15s %7d %7d %.5f\n'
#         for i in range(num_class):
#             fd.write(format_str % (class_list[i], result_ll[i][0], result_ll[i][1], accuracy[i]))
#         fd.write(format_str % ('(Total)', correct_t, incorrect_t, accuracy_t))
#         # tp, fp, tn, fn
#         result_ll = [[0 for __ in range(4)] for _ in range(num_attr)]
#         idx = 0
#         while idx < total_cnt:
#             print('Current Idx: '  + str(idx))
#             end = min([idx + batch_size, total_cnt])
#             this_batch_size = end - idx
#
#             probs_val, labels_val = sess.run([test_probs, test_labels])
#             preds = np.around(probs_val)
#             for i in range(this_batch_size):
#                 for j in range(num_attr):
#                     if labels_val[i, j] == 0:
#                         if preds[i, j] == 0: # tn
#                             result_ll[j][2] += 1
#                         if preds[i, j] == 1: # fp
#                             result_ll[j][1] += 1
#                     if labels_val[i, j] == 1:
#                         if preds[i, j] == 0: # fn
#                             result_ll[j][3] += 1
#                         if preds[i, j] == 1: # tp
#                             result_ll[j][0] += 1
#             idx = end
#
#         coord.request_stop()
#         coord.wait_for_stop(1)
#         sess.close()
#
#
#     accuracy, precision, recall, f1 = ([], [], [], [])
#     for i in range(num_attr):
#         tp, fp, tn, fn = result_ll[i]
#         assert total_cnt == (tp+fp+tn+fn)
#         accuracy.append((tp+tn)/float(total_cnt))
#         if tp+fp == 0:
#             precision.append(0.0)
#         else:
#             precision.append(tp/float(tp+fp))
#         if tp+fn == 0:
#             recall.append(0.0)
#         else:
#             recall.append(tp/float(tp+fn))
#         if precision[i] == .0 or np.isnan(precision[i]) or recall[i] == .0:
#             f1.append(0.0)
#         else:
#             f1.append(2/(1/precision[i]+1/recall[i]))
#
#     result_t = [0 for _ in range(4)]
#     for i in range(num_attr):
#         for j in range(4):
#             result_t[j] += result_ll[i][j]
#     tp_t, fp_t, tn_t, fn_t = result_t
# #    accuracy_t = float(tp_t+tn_t)/float(sum(result_t))
# #    precision_t = tp_t/float(tp_t+fp_t)
# #    recall_t = tp_t/float(tp_t+fn_t)
# #    f1_t = 2./(1./precision_t+1./recall_t)
#     accuracy_t = np.average(accuracy)
#     precision_t = np.average(precision)
#     recall_t = np.average(recall)
#     f1_t = np.average(f1)
#
#     print 'Attribute\tTP\tFP\tTN\tFN\tAcc.\tPrec.\tRecall\tF1-score'
#     for i in range(num_attr):
#         tp, fp, tn, fn = result_ll[i]
#         format_str = '%-15s %7d %7d %7d %7d %.5f %.5f %.5f %.5f'
#         print format_str % (attribute_list[i].replace(' ', '-'),tp,fp,tn,fn,accuracy[i],precision[i],recall[i],f1[i])
#     print(format_str % ('(Total)',tp_t,fp_t,tn_t,fn_t,accuracy_t,precision_t,recall_t,f1_t))
#     with open(FLAGS.output, 'w') as fd:
#         fd.write('Attribute\tTP\tFP\tTN\tFN\tAcc.\tPrec.\tRecall\tF1-score\n')
#         for i in range(num_attr):
#             tp, fp, tn, fn = result_ll[i]
#             format_str = '%-15s %7d %7d %7d %7d %.5f %.5f %.5f %.5f\n'
#             fd.write(format_str % (attribute_list[i].replace(' ', '-'),tp,fp,tn,fn,accuracy[i],precision[i],recall[i],f1[i]))
#         fd.write(format_str % ('(Total)',tp_t,fp_t,tn_t,fn_t,accuracy_t,precision_t,recall_t,f1_t))

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
