from __future__ import division
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from nets import inception_utils, inception_v3
from glob import glob
import imageio
import sys
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

def get_checkpoint_init_fn():
    # Load from .ckpt file
    variables_to_restore = slim.get_variables_to_restore(exclude=["InceptionV3/Logits/Conv2d_1c_1x1/weights:0", "InceptionV3/Logits/Conv2d_1c_1x1/biases:0"])
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    slim_init_fn = slim.assign_from_checkpoint_fn("./inception_v3.ckpt",variables_to_restore,ignore_missing_vars=True)
    return slim_init_fn

df = pd.read_csv("./train_labels.csv")

df_zero_train = df[df["label"]==0].iloc[:40000]
df_one_train = df[df["label"]==1].iloc[:40000]

df_zero_val = df[df["label"]==0].iloc[40000:44000]
df_one_val = df[df["label"]==1].iloc[40000:44000]

# Load Data

slim = tf.contrib.slim
arg_scope = tf.contrib.framework.arg_scope

# Set Params
MAX_EPOCH = 150000
NUM_CLASSES = 2
NUM_IMG_FROM_EACH_CLASS = 50
input_size = NUM_IMG_FROM_EACH_CLASS * NUM_CLASSES
VALIDATION_INTERVAL = 500
START_LR = 1e-04
DECAY_STEP = 10000 / 63 * 10
DECAY_RATE = 0.98
RETRAIN_NAMES = ["Logits", "Mixed_7"]

if sys.platform == "darwin":
    LOG_DIR = "./saved_model/Inception_" + str(START_LR) + "_" + str(DECAY_STEP) + "_" + str(DECAY_RATE) + "_" + RETRAIN_NAMES[-1]
else:
    LOG_DIR = "./saved_model/Inception_" + str(START_LR) + "_" + str(DECAY_STEP) + "_" + str(DECAY_RATE) + "_" + RETRAIN_NAMES[-1]


session_config = tf.ConfigProto(log_device_placement=False)
session_config.gpu_options.allow_growth = True

# Define Graph
one_hot_encoder = OneHotEncoder(2)
one_hot_encoder.fit(np.arange(2).reshape(-1,1))

g = tf.Graph()

with g.as_default():
    img_holder = tf.placeholder(shape=[input_size,96,96,3], dtype=tf.float32, name="Img_Holder")
    label_holder = tf.placeholder(shape=[input_size,2], dtype=tf.float32, name="Label_Holder")
    img = tf.Variable(img_holder, name="Img_Var", trainable=False)
    label = tf.Variable(label_holder, name="Label_Var", trainable=False)
    img_assign = img.assign(img_holder, name="Img_Assign")
    label_assign = label.assign(label_holder, name="Label_Assign")
    
    img_holder_val = tf.placeholder(shape=[input_size,96,96,3], dtype=tf.float32, name="Img_Holder_val")
    label_holder_val = tf.placeholder(shape=[input_size,2], dtype=tf.float32, name="Label_Holder_val")
    img_val = tf.Variable(img_holder_val, name="Img_Var_val", trainable=False)
    label_val = tf.Variable(label_holder_val, name="Label_Var_val", trainable=False)
    img_assign_val = img_val.assign(img_holder_val, name="Img_Assign_val")
    label_assign_val = label_val.assign(label_holder_val, name="Label_Assign_val")

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_points = inception_v3.inception_v3(img, num_classes=2, create_aux_logits=False, is_training=True)
        pred = slim.softmax(logits, scope="Prediction")
        logits_val, _ = inception_v3.inception_v3(img_val, num_classes=2, create_aux_logits=False, is_training=False, reuse=tf.AUTO_REUSE)
        pred_val = slim.softmax(logits_val, scope="Prediction_Val")
    _, accuracy_val = tf.metrics.accuracy(tf.math.argmax(pred_val, axis=1), tf.math.argmax(label_val, axis=1))
    
    loss = tf.losses.softmax_cross_entropy(label, logits)
    total_loss = tf.losses.get_total_loss()
    loss_val = tf.losses.softmax_cross_entropy(label_val, logits_val, loss_collection="validation")
    
    retrain_list = []
    for v in tf.trainable_variables():
        for n in RETRAIN_NAMES:
            if n in v.name:
                retrain_list += [v]
    
    learning_rate = tf.train.exponential_decay(START_LR, tf.train.get_or_create_global_step(), DECAY_STEP, DECAY_RATE)
    opt = tf.train.AdamOptimizer(learning_rate)
    train_tensor = slim.learning.create_train_op(total_loss, optimizer=opt, variables_to_train=retrain_list)
    # Creat Summary
    slim.summaries.add_scalar_summary(total_loss, 'cross_entropy_loss', 'losses')
    slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
    slim.summaries.add_scalar_summary(loss_val, 'validation_loss', 'losses')
    slim.summaries.add_scalar_summary(loss_val-total_loss, 'validation_delta', 'losses')
    slim.summaries.add_scalar_summary(accuracy_val, 'validation_accuracy', 'accuracy')
    
def train_step_fn(sess, train_op, global_step, train_step_kwargs):
    """
    slim.learning.train_step():
    train_step_kwargs = {summary_writer:, should_log:, should_stop:}
    """
#     train_step_fn.step += 1  # or use global_step.eval(session=sess)
    input_df = [df_zero_train.sample(NUM_IMG_FROM_EACH_CLASS), df_one_train.sample(NUM_IMG_FROM_EACH_CLASS)]
    input_path = np.array([input_df[i]["id"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_images = np.array([imageio.imread("./train/"+i+".tif") for i in input_path]).astype(np.float32)

    labels = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)

    input_images, labels = shuffle(input_images, labels)
    labels = one_hot_encoder.transform(labels).toarray()
    
    sess.run([img_assign,label_assign], feed_dict={img_holder:input_images, label_holder:labels})
#     print sess.run([img,label])

    # calc training losses
    total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)


    # validate on interval
    if global_step.eval(session=sess) % VALIDATION_INTERVAL == 0:
        input_df_val = [df_zero_val.sample(NUM_IMG_FROM_EACH_CLASS), df_one_val.sample(NUM_IMG_FROM_EACH_CLASS)]
        input_path_val = np.array([input_df_val[i]["id"].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]).reshape(-1)
        input_images_val = np.array([imageio.imread("./train/"+i+".tif") for i in input_path_val]).astype(np.float32)

        labels_val = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)

        input_images_val, labels_val = shuffle(input_images_val, labels_val)
        labels_val = one_hot_encoder.transform(labels_val).toarray()
        
        sess.run([img_assign_val,label_assign_val,logits_val,accuracy_val], 
                 feed_dict={img_holder_val:input_images_val, label_holder_val:labels_val})
        validiate_loss = sess.run(loss_val)
    
        
#    print(">> global step {}:    train={}   validation={}  delta={}".format(global_step.eval(session=sess), 
#                        total_loss, loss_val, loss_val-total_loss))


    return [total_loss, should_stop]

with g.as_default():
    
    # Train Set    
    input_df = [df_zero_train.sample(NUM_IMG_FROM_EACH_CLASS), df_one_train.sample(NUM_IMG_FROM_EACH_CLASS)]
    input_path = np.array([input_df[i]["id"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_images = np.array([imageio.imread("./train/"+i+".tif") for i in input_path]).astype(np.float32)

    labels = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)

    input_images, labels = shuffle(input_images, labels)
    labels = one_hot_encoder.transform(labels).toarray()
    
    # Val Set
    input_df_val = [df_zero_val.sample(NUM_IMG_FROM_EACH_CLASS), df_one_val.sample(NUM_IMG_FROM_EACH_CLASS)]
    input_path_val = np.array([input_df_val[i]["id"].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]).reshape(-1)
    input_images_val = np.array([imageio.imread("./train/"+i+".tif") for i in input_path_val]).astype(np.float32)

    labels_val = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)

    input_images_val, labels_val = shuffle(input_images_val, labels_val)
    labels_val = one_hot_encoder.transform(labels_val).toarray()

    slim.learning.train(
        train_tensor,
        LOG_DIR,
        log_every_n_steps=1,
        number_of_steps=MAX_EPOCH,
        graph=g,
        save_summaries_secs=60,
        save_interval_secs=300,
        init_fn=get_checkpoint_init_fn(),
        global_step=tf.train.get_global_step(),
        train_step_fn = train_step_fn,
        session_config=session_config,
        init_feed_dict = {img_holder:input_images, label_holder:labels, img_holder_val: input_images_val, label_holder_val: labels_val})
