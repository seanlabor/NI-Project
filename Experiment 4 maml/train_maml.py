import numpy as np 
import tensorflow as tf # use the tf2.4
import tensorflow_datasets as tfds 
import tensorflow_addons as tfa
from adabelief_tf import AdaBeliefOptimizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

from torch.utils.tensorboard import SummaryWriter


inner_lr = 1E-5
outter_lr = 1E-5
inner_task_loop_no = 64 # give 32 tasks of 2ways-1shot
# opt_inner = tf.keras.optimizers.Adagrad(inner_lr)
# opt_outter = tf.keras.optimizers.Adagrad(outter_lr, clipnorm=1.)
opt_inner = tfa.optimizers.NovoGrad(inner_lr)
opt_outter = tfa.optimizers.NovoGrad(outter_lr, clipnorm=1.)
# opt_inner = tfa.optimizers.RectifiedAdam(inner_lr)
# opt_outter = tfa.optimizers.RectifiedAdam(outter_lr, clipnorm=1.)
# opt_inner = AdaBeliefOptimizer(inner_lr)
# opt_outter = AdaBeliefOptimizer(outter_lr)
# opt_inner = tf.keras.optimizers.SGD(inner_lr, momentum=.0)
# opt_outter = tf.keras.optimizers.SGD(outter_lr, momentum=.9, clipnorm=1.)

#opt_outter = tfa.optimizers.MovingAverage(opt_outter) # using average strategy in tfa

def cnn():
    Input = tf.keras.Input([28, 28, 1])
    Input_n = Input/128.0 - 1
    conv1 = tf.keras.layers.Conv2D(32, [3, 3], strides=(2, 2), padding="SAME", kernel_initializer="he_uniform", activation=tf.nn.relu)(Input_n) #[14,14]
    conv2 = tf.keras.layers.Conv2D(64, [3, 3], strides=(2, 2), padding="SAME", kernel_initializer="he_uniform", activation=tf.nn.relu)(conv1) #[7,7]
    conv3 = tf.keras.layers.Conv2D(128, [3, 3], strides=(2, 2), padding="SAME", kernel_initializer="he_uniform", activation=tf.nn.relu)(conv2) #[4,4]
    fc = tf.keras.layers.Flatten()(conv3)
    fc1 = tf.keras.layers.Dense(128, kernel_initializer="he_uniform", activation=tf.nn.relu)(fc)
    fc2 = tf.keras.layers.Dense(256, kernel_initializer="he_uniform", activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(512, kernel_initializer="he_uniform", activation=tf.nn.relu)(fc2)
    out = tf.keras.layers.Dense(1, kernel_initializer="he_uniform", activation=None)(fc3)
    return tf.keras.Model(inputs=Input, outputs=out)
pass 

def select_support_query_set(dataset_iter):
    flag_1 = flag_0 = True
    imgs_1_support, labs_1_support = next(dataset_iter)
    imgs_0_support, labs_0_support = next(dataset_iter)
    support_set = tf.concat([imgs_1_support, imgs_0_support], axis=0)
    while(flag_1 or flag_0):
        imgs_query, labs_query = next(dataset_iter)
        if ((labs_query.numpy() == labs_1_support.numpy()) and flag_1) :
            imgs_1_query = tf.Variable(imgs_query)
            flag_1 = False
        elif ((labs_query.numpy() == labs_0_support.numpy()) and flag_0) :
            imgs_0_query = tf.Variable(imgs_query)
            flag_0 = False
        pass 
    pass 
    query_set = tf.concat([imgs_1_query, imgs_0_query], axis=0)
    return [support_set, query_set]
pass 

(kmnist_tr, kmnist_ts) = tfds.load('kmnist', 
                                   split=['train','test'],
                                   shuffle_files=True,
                                   as_supervised=True,
                                   with_info=False,)

kmnist_tr = kmnist_tr.apply(lambda ds: ds.map(lambda x, y: (x, y + 60)))
kmnist_ts = kmnist_ts.apply(lambda ds: ds.map(lambda x, y: (x, y + 60)))

##todo: filter einbauen



(emnist_tr, emnist_ts) = tfds.load('emnist', 
                                   split=['train','test'],
                                   shuffle_files=True,
                                   as_supervised=True,
                                   with_info=False,)

kmnist_tr = kmnist_tr.concatenate(emnist_tr).shuffle(100000)
kmnist_ts = kmnist_ts.concatenate(emnist_ts).shuffle(100000)


def out_of_distro_classes(x,y):
    i_y = int(y)
    return y == 60 or  y == 61 or y == 5 or y == 14 or y == 17
in_of_distro_classes = lambda x, y: not out_of_distro_classes(x, y)

kmnist_tr_out_of_distro_classes = kmnist_tr.filter(out_of_distro_classes)
kminst_tr_in_of_distro_classes = kmnist_tr.filter(in_of_distro_classes)


kmnist_ts_out_of_distro_classes = kmnist_ts.filter(out_of_distro_classes)
#kminst_ts_in_of_distro_classes = kmnist_ts.filter(in_of_distro_classes)





# Task hint:
# this example will use 2ways-1shot for training the MAML.
# the inner task loop will set as 30. 
kmnist_tr = kmnist_tr.batch(1).repeat()
kmnist_tr_iter = iter(kmnist_tr)
meta_labs = tf.Variable([[1], [0]], dtype=tf.float32)

cnn_model = cnn()

# according to the TF2 manual, the variable of the model would need to be
# initialized, and use it to inferece something is a way to initialize 
# the varialbe.
support_set, query_set = select_support_query_set(kmnist_tr_iter)
_ = cnn_model(support_set) # 1)for initializing the model. 2)for define the loss function out-of the loop 
loss_inner_meta = lambda: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=meta_labs, logits=cnn_model(support_set)))
loss_inner_task = lambda: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=meta_labs, logits=cnn_model(query_set)))


def eval_model():
    

    ## multiclass binäre Operatoren

    which_class = 5
    potenzielle_chars = list()
    nicht_potenzielle_chars = list()
    limit = 2
    for j, (x, y) in enumerate(kmnist_tr):
        if y == which_class:
            potenzielle_chars.append((x, y))
        elif len(nicht_potenzielle_chars) <= limit:
            nicht_potenzielle_chars.append([x, y])

        if len(potenzielle_chars) > limit and len(nicht_potenzielle_chars) > limit:
            break
            
    clone_cnn_model = tf.keras.models.clone_model(cnn_model)
    clone_cnn_model.compile( tf.keras.optimizers.SGD() , tf.keras.losses.BinaryCrossentropy(from_logits=True))

    print("train_inner_model")
    for x, y in potenzielle_chars:
        clone_cnn_model.fit(x, tf.ones([1, 1]))
    for x, y in nicht_potenzielle_chars:
        clone_cnn_model.fit(x, tf.zeros([1, 1]))
    pred_list = list()
    true_list = list()

    roc_i = 0
    for x, y in kmnist_ts:
        if int(y) == which_class:
            roc_i += 1
        true_list.append(int(y) == which_class)
        pred = clone_cnn_model(x.numpy().reshape([1, 28, 28, 1]))
        preds = tf.math.sigmoid(pred)
        pred_list.append(preds[0])

        if roc_i > 5:
            break
    return roc_auc_score(true_list, pred_list)

        
summary_writer = SummaryWriter()
for step in range(50000):
    # keep the original weights for meta-training
    meta_weights = [tf.Variable(target_weights) for target_weights in cnn_model.trainable_weights]
    
    def meta_loss():
        loss_outter = 0
        for task_count in range(inner_task_loop_no):
            # fetch the dataset. The first dataset is always be 1. 
            # So, the labels will be defined as the meta-labels insteadly. 
            support_set, query_set = select_support_query_set(kmnist_tr_iter)
            
            # put the meta-weights into the model
            for model_weight_index in range(len(cnn_model.trainable_weights)):
                cnn_model.trainable_weights[model_weight_index].assign(meta_weights[model_weight_index])
            pass

            # updateing the weights from meta-weights
            opt_inner.minimize(loss=loss_inner_meta, var_list=cnn_model.trainable_weights) 
            
            # calculating the task loss (after the meta-weights updating)
            # here, the same support set and query set are used. you can 
            # also creat a different query set for meta-weight training.
            loss_outter += loss_inner_task()
        pass 
        return loss_outter/inner_task_loop_no
    pass 

    # put the meta-weights into the model
    for model_weight_index in range(len(cnn_model.trainable_weights)):
        cnn_model.trainable_weights[model_weight_index].assign(meta_weights[model_weight_index])
    pass

    # update the meta-weights
    opt_outter.minimize(loss=meta_loss, var_list=cnn_model.trainable_weights) 

    print("step: {}  outter_loss: {}".format(step, meta_loss()))
    summary_writer.add_scalar("train/loss", meta_loss().numpy(), step)
    
    roc_auc = eval_model()
    roc_auc = roc_auc if roc_auc > 0.5 else 1 - roc_auc
    print(roc_auc)
    summary_writer.add_scalar("test/roc_auc", roc_auc, step)

