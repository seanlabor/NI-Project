import numpy as np 
import tensorflow as tf # use the tf2.4
import tensorflow_datasets as tfds 
import tensorflow_addons as tfa
from adabelief_tf import AdaBeliefOptimizer
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from random import shuffle

from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import time

inner_lr = 1E-5
outter_lr = 1E-5
inner_task_loop_no = 64 # give 32 tasks of 2ways-1shot
# opt_inner = tf.keras.optimizers.Adagrad(inner_lr)
# opt_outter = tf.keras.optimizers.Adagrad(outter_lr, clipnorm=1.)
opt_inner = tfa.optimizers.NovoGrad(inner_lr)
opt_outter = tfa.optimizers.NovoGrad(outter_lr, clipnorm=1.)
#for meaning of the modes check here https://scikit-learn.org/stable/modules/multiclass.html
n_first_training = 3
n_second_training = 3


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



kmnist_tr_ns = kmnist_tr.apply(lambda ds: ds.map(
        lambda x, y: 
            (tf.reshape(x, [1, 28, 28, 1]), y)
        ))
kmnist_ts_ns = kmnist_ts.apply(lambda ds: ds.map(
        lambda x, y: 
            (tf.reshape(x, [1, 28, 28, 1]), y)
        ))

out_of_distro_classes_list = [60, 61, 5, 14, 17]
out_of_distro_tr = dict()
out_of_distro_ts = dict()
in_of_distro_tr = dict()



#implementiere Shuffle hier.


#collects all classes for out_of_distro_tr
for ofd in out_of_distro_classes_list:
    out_of_distro_tr[ofd] = kmnist_tr_ns.filter(lambda x, y: int(y) == ofd)

#collects all classes for out_of_distro_ts
for ofd in out_of_distro_classes_list:
    out_of_distro_ts[ofd] = kmnist_ts_ns.filter(lambda x, y: int(y) == ofd)

    
# collects all opposite classes for in_of_distro_tr
for ofd in out_of_distro_classes_list:
    is_first_time = True
    for ofd_inner in out_of_distro_classes_list:
        if ofd_inner == ofd:
            continue
        if is_first_time:
            in_of_distro_tr[ofd] = out_of_distro_tr[ofd_inner]
            is_first_time = False
        else:
            in_of_distro_tr[ofd] = in_of_distro_tr[ofd].concatenate(out_of_distro_tr[ofd_inner])



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


def eval_model(mode_binary_classifier):
    #https://scikit-learn.org/stable/modules/multiclass.html
    #https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

    #mode_binary_classifier = "OneVsRest" #OneVsRest,  OneVsOne


    if mode_binary_classifier == "OneVsRest":
        cloned_models = dict()
        for ofd in out_of_distro_classes_list:
            cloned_models[ofd] = tf.keras.models.clone_model(cnn_model)
            cloned_models[ofd].compile( tf.keras.optimizers.SGD(), tf.keras.losses.BinaryCrossentropy(from_logits=True))

            training_elements = list()

            for i, (x, _) in enumerate(out_of_distro_tr[ofd]):
                if i >= n_first_training:
                    break
                training_elements.append((x, tf.ones([1,1])))

            for i, (x, _) in enumerate(in_of_distro_tr[ofd]):
                if i >= n_second_training:
                    break
                training_elements.append((x, tf.zeros([1, 1])))

            shuffle(training_elements)
            for x, y in training_elements:
                cloned_models[ofd].fit(x, y)

        list_pred_classes = list()
        list_true_classes = list()
        for i, (x, y) in enumerate(kmnist_ts_ns):
            if i > 100:
                break
            predictions = list()
            for ofd in out_of_distro_classes_list:
                pred_ofd = cloned_models[ofd](x)
                predictions.append((ofd, pred_ofd))

            max_pred_class, _ = max(predictions, key=lambda x: x[1])
            list_true_classes.append(int(y))
            list_pred_classes.append(max_pred_class)
        
        accuracy = accuracy_score(list_true_classes, list_pred_classes)





    elif mode_binary_classifier == "OneVsOne":
        used_pairs = list()
        cloned_models = dict()

        for ofd_outer in out_of_distro_classes_list:
            for ofd_inner in out_of_distro_classes_list:
                if (ofd_outer, ofd_inner) in used_pairs or \
                    (ofd_inner, ofd_outer) in used_pairs or \
                    ofd_inner == ofd_outer:
                    break
                
                used_pairs.append((ofd_outer, ofd_inner))
                training_elements = list()

                for i, (x, _) in enumerate(out_of_distro_tr[ofd_outer]):
                    if i >= n_first_training:
                        break
                    training_elements.append((x, tf.ones([1,1])))

                for i, (x, _) in enumerate(in_of_distro_tr[ofd_inner]):
                    if i >= n_second_training:
                        break
                    training_elements.append((x, tf.zeros([1, 1])))

                cloned_models[(ofd_outer, ofd_inner)] = tf.keras.models.clone_model(cnn_model)
                cloned_models[(ofd_outer, ofd_inner)].compile( 
                    tf.keras.optimizers.SGD(), 
                    tf.keras.losses.BinaryCrossentropy(from_logits=True)
                    )
                
                shuffle(training_elements)

                for x, y in training_elements:
                    cloned_models[(ofd_outer, ofd_inner)].fit(x, y)

        list_pred_classes = list()
        list_true_classes = list()

        for i, (x, y) in enumerate(kmnist_ts_ns):
            if i > 100:
                break

            predictions_votes = list()

            for ofd_outter, ofd_inner in cloned_models:
                pred = cloned_models[(ofd_outter, ofd_inner)](x)
                if pred >= 0.5:
                    predictions_votes.append(ofd_outter)
                else:
                    predictions_votes.append(ofd_inner)

            ##calculates, which vote / elemenst is most frequent
            prediction_counter = Counter(predictions_votes)
            most_common_vote = prediction_counter.most_common(1)[0][0]

            list_pred_classes.append(most_common_vote)
            list_true_classes.append(int(y))

        accuracy = accuracy_score(list_true_classes, list_pred_classes)

            



    else:
        raise Exception("invalid mode_binary_classifier")

    return accuracy

        
summary_writer = SummaryWriter(log_dir=f"run/{time.time()}_{n_first_training=}_{n_second_training=}")
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
    
    
    acc_onevsrest = eval_model("OneVsRest")
    acc_onevsone = eval_model("OneVsOne")
    print(f"acc OneVsRest: {acc_onevsrest} \t OneVsOne {acc_onevsone}")
    summary_writer.add_scalar("test/acc_one_vs_rest", acc_onevsrest, step)
    summary_writer.add_scalar("test/acc_one_vs_one", acc_onevsone, step)


