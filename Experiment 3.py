import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
import _pickle as cPickle
import pickle
import copy
from jax import jit, vmap, pmap, grad, value_and_grad
import random
from torchvision.datasets import MNIST
from torchvision.datasets import KMNIST
from torchvision.datasets import CIFAR100
from torchvision.datasets import EMNIST
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from jax.example_libraries import stax, optimizers
import torchvision
import torch
from sklearn.neighbors import NearestNeighbors
import torch.utils.data as data_utils
from jax.flatten_util import ravel_pytree
import os
import time
import shutil
import _pickle as cPickle
import time
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from sklearn.model_selection import train_test_split
from jax import random, value_and_grad
import haiku as hk
from math import isnan, isinf
from torch.utils.tensorboard import SummaryWriter
from statistics import mean, stdev, median

#Needs Cleaning

'''Set your file directorys'''

googledrive_path="/content/drive/MyDrive/Colab Notebooks/Jax_MNist/"
##local_path="C:/Users/Flo/Documents/Uni/Masterarbeit/Hanabi/Mnist handwritten digits"
local_path="/vol/fob-vol3/mi20/kralaben/Dokumente/NIProject"





'''number of training epochs for Network2'''
n_training_epochs = 3 #number of training epochs for every NN2.  
n_offsp_epoch = 10 #number of training and testing runs combined, to get an average for the performance of the Convu net.
n_testing_epochs = 3 # number of testing runs, per n_offsp_epoch

'''parameter Network2'''
n_samples = 150  #Number of training samples for NN2, distribution of data for 600: [68, 59, 66, 67, 47, 46, 65, 71, 53, 58])
n_test=1000 #Number of test samples for NN2. Needs to be multiples of batch_size_test=500 )
batch_size = 50 # for precise n_samples number must be: n_samples%batch_size_train=0

learning_rate = 0.1
momentum = 0.5
log_interval = 10



'''Standard deviation for gaussian noise in Network1'''
'''!!! Important hyperparameter, >=1 gives bad results'''
std_modifier=0.05



'''number of offsprings per metaepoch'''
n_offsprings=100
'''number of metaopochs'''
n_metaepochs=10


NNin1=2500 #dependent on Convu
NNout1=10



'''Convunet'''
Convu1_in=1
Convu2_in=12
Convu3_in=24
seed_convu=0

n_samples=150 #number of training samples
batch_size = 50
n_test=1000
n_training_epochs=3
print_distribution_data=False
std_modifier=0.05


Convu1_in=32
Convu2_in=16
Convu3_in=4
kernelsize_=(3,3)



use_sigma_decay=True
sigma_start=1.0
sigma_goal=0.1

'''logging to screen variables'''
print_offsprings=True
print_distribution_data=False
use_sigma_decay=True #otherwise using constant sigma from config tab
sigma_start=1 
sigma_goal=0.05 #sigma goal after n_metaepochs
        
'''choose either method, softmax or elitist=keep only best offspring'''
use_softmax=True
temperature=0.05
use_elitist=False
use_winnerlist=False

n_metaepochs=30 #overwriting variable from config tab, delete later
n_offsprings=10 #overwriting variable from config tab, delete later

'''number of training epochs for Network2'''
n_offsp_epoch = 2
n_testing_epochs = 5
n_metaepochs=10



def logg_to_file (string_, array=None):
    if array is None:

        file1 = open(save_txt,"a")
        file1.write(string_)
        file1.write("\n")
        file1.close()
    
    if array is not None:

        file1 = open(save_txt,"a")
        file1.write(string_)
        file1.write(str(array))
        file1.write("\n")
        file1.close()

def log_variables():
    
    logg_to_file (("n_training_epochs = {}".format(n_training_epochs)))
    logg_to_file (("n_offsp_epoch = {}".format(n_offsp_epoch)))
    
    logg_to_file (("n_samples = {}".format(n_samples)))
    logg_to_file (("n_test = {}".format(n_test)))
    logg_to_file (("batch_size = {}".format(batch_size)))

    logg_to_file (("use_focus = {}".format(use_focus)))
    #logg_to_file (("focus_layer = {}".format(focus_layer)))
    logg_to_file (("focus_change_every = {}".format(focus_change_every)))

    
    logg_to_file (("use_sigma_decay = {}".format(use_sigma_decay)))
    logg_to_file (("n_decay_epochs = {}".format(n_decay_epochs)))
    logg_to_file (("sigma_start = {}".format(sigma_start)))
    logg_to_file (("sigma_goal = {}".format(sigma_goal)))

  
    logg_to_file (("use_KNN = {}".format(use_KNN)))
    logg_to_file (("KNN_n_neighbors = {}".format(KNN_n_neighbors)))
    logg_to_file (("KNN_top_n = {}".format(KNN_top_n)))
    logg_to_file (("n_KNN_subsprings = {}".format(n_KNN_subsprings)))




    logg_to_file (("std_modifier = {}".format(std_modifier)))
    logg_to_file (("use_sigma_decay = {}".format(use_sigma_decay)))
    logg_to_file (("sigma_start = {}".format(sigma_start)))
    logg_to_file (("sigma_goal = {}".format(sigma_goal)))
    logg_to_file (("n_decay_epochs = {}".format(n_decay_epochs)))
    logg_to_file (("use_pickle = {}".format(use_pickle)))
    logg_to_file (("pickle_path = {}".format(pickle_path)))
    logg_to_file (("use_father = {}".format(use_father)))


    logg_to_file (("NNin1 = {}".format(NNin1)))
    logg_to_file (("NNout1 = {}".format(NNout1)))
    logg_to_file (("Convu_in1 = {}".format(Convu1_in)))
    logg_to_file (("Convu2_in = {}".format(Convu2_in)))
    logg_to_file (("Convu3_in = {}".format(Convu3_in)))

    logg_to_file (("kernelsize_ = {}".format(kernelsize_)))
    
    logg_to_file (("n_metaepochs = {}".format(n_metaepochs)))
    logg_to_file (("n_testing_epochs = {}".format(n_testing_epochs)))         
    logg_to_file (("n_offsp_epoch = {}".format(n_offsp_epoch)))
    logg_to_file (("n_offsprings = {}".format(n_offsprings)))

    logg_to_file (("use_softmax = {}".format(use_softmax)))
    logg_to_file (("temperature = {}".format(temperature)))



def pathandstuff():

    global save_txt
    global base_path
    global save_path

    if os.path.exists(local_path):
        '''Save running code file to log folder'''
        #nb_full_path = os.path.join(os.getcwd(), nb_name) #path of current notebook
        #shutil.copy2(nb_full_path, save_path) #save running code file to log folder
        print("on local")
        base_path=local_path
    elif os.path.exists(googledrive_path):
        print("on google")
        base_path=googledrive_path
    else:
        raise ValueError('Please specify save path or connect to Google Drive')
        
    logs_path=base_path+"Logs/"
    '''Set logging and temp paths'''
    timestamp=time.strftime("%d.%m.%Y_%H.%M")
    foldername=timestamp
    save_path=os.path.join(logs_path,foldername,)
    save_path=save_path+"/"
    save_txt = os.path.join(save_path, 'Log_Jax_MNist_{}.txt'.format(foldername))

    print("Log path:",save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)



def logg_script(file_name, save_path):
    source=f"{save_path}logs/"
    os.makedirs(source)
    destination=save_path+f"{file_name}.ipynb"
    shutil.copy2(source, destination)

'''logging to txt and print'''
def logg (string_, array=None):
    if array is None:

        file1 = open(save_txt,"a")
        file1.write(string_)
        file1.write("\n")
        file1.close()
        print(string_)
    if array is not None:

        file1 = open(save_txt,"a")
        file1.write(string_)
        file1.write(str(array))
        file1.write("\n")
        file1.close()
        print(string_, array)
        
"""
train_dataset = MNIST(root='train_mnist', train=True, download=True,transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

test_dataset = MNIST(root='test_mnist', train=False, download=True,transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

x = np.concatenate((train_dataset.data,test_dataset.data))
y= np.concatenate((train_dataset.targets,test_dataset.targets))

x = jnp.array(x,dtype="float32").reshape(len(x), -1)
y = jnp.array(y)
"""



data_image_size=28
data_image_depth=1

train_dataset_EMNIST = EMNIST(root='train_emnist', train=True, download=True,split="balanced", transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ]))

test_dataset_EMNIST = EMNIST(root='test_emnist', train=False, download=True,split="balanced",transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

train_dataset_KMNIST = KMNIST(root='train_kmnist', train=True, download=True,transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ]))

test_dataset_KMNIST = KMNIST(root='test_kmnist', train=False, download=True,transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

'''Relabel KMNIST to avoid overlaps'''
map_dic = dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [47,48,49,50,51,52,53,54,55,56]))
train_dataset_KMNIST.targets = np.vectorize(map_dic.get)(np.array(train_dataset_KMNIST.targets))
train_dataset_KMNIST.targets = t = torch.from_numpy(train_dataset_KMNIST.targets)


test_dataset_KMNIST.targets = np.vectorize(map_dic.get)(np.array(test_dataset_KMNIST.targets))
test_dataset_KMNIST.targets = t = torch.from_numpy(test_dataset_KMNIST.targets)


x = np.concatenate((train_dataset_EMNIST.data,
                    test_dataset_EMNIST.data,
                    train_dataset_KMNIST.data,
                    test_dataset_KMNIST.data
                )).astype(np.float32)

y= np.concatenate((train_dataset_EMNIST.targets,
                   test_dataset_EMNIST.targets,
                   train_dataset_KMNIST.targets,
                   test_dataset_KMNIST.targets
                   )).astype(np.float32)

print("Number of different classes:", len(list(set(list(np.array(y))))))

def init_MLP(layer_widths, parent_key, scale=0.01):

    params = []
    keys = jax.random.split(parent_key, num=len(layer_widths)-1) #

    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = jax.random.split(key)
        params.append([
                       scale*jax.random.normal(weight_key, shape=(out_width, in_width)),
                       scale*jax.random.normal(bias_key, shape=(out_width,))
                       ]
        )
    return params



@jit
def MLP_predict(params, x):

    hidden_layers = params[:-1]
    activation = x

    for w, b in hidden_layers:
        activation = jax.nn.relu(jnp.dot(w, activation) + b)

    w_last, b_last = params[-1]
    logits = jnp.dot(w_last, activation) + b_last

    return logits - logsumexp(logits)

jit_MLP_predict=jit(MLP_predict)


@jit
def batched_MLP_predict(params,x):
    return vmap(jit_MLP_predict, (None, 0))(params,x)
  
jit_batched_MLP_predict=jit(batched_MLP_predict)


Convu1_in=16
Convu2_in=24
Convu3_in=1

conv_init, conv_apply = stax.serial(
    stax.Conv(Convu1_in,kernelsize_, padding="SAME"),
    stax.BatchNorm(),
    stax.Relu,
    stax.MaxPool((2,2)),
    stax.Conv(Convu2_in, kernelsize_, padding="SAME"),
    stax.BatchNorm(),
    stax.Relu,
    stax.MaxPool((2,2)),
    stax.Conv(Convu3_in, kernelsize_, padding="SAME"),
    stax.Relu,
    stax.MaxPool((2,2))
)


'''After changing Convu structure test if convu out and NN in matches, set NNin1=25*25*4 to corresponding shape in error (5, 25, 25, 4) '''
NNin1=625
rng=jax.random.PRNGKey(1)

father_weights = conv_init(rng, (batch_size,28,28,1))
father_weights = father_weights[1]


x_train=x[random.randint(rng, (n_offsp_epoch*n_samples,), 0, 60000, dtype='uint8')]
testaffe=x_train[0:5]
imgs = conv_apply(father_weights, testaffe.reshape(-1,28,28,1))

MLP_params = init_MLP([NNin1, 10], rng)

pred_classes = jnp.argmax(jit_batched_MLP_predict(MLP_params, imgs.reshape(-1,NNin1)), axis=1)


@jit
def loss_fn(params, imgs, gt_lbls):
  
    predictions = jit_batched_MLP_predict(params, imgs)
    #print("predictions",predictions.shape)
    return -jnp.mean(predictions * gt_lbls)
    
jit_loss_fn=jit(loss_fn)

@jit
def update(params, imgs, gt_lbls, lr=0.01):
    loss, grads = value_and_grad(loss_fn)(params, imgs, gt_lbls)

    return loss, jax.tree_multimap(lambda p, g: p - lr*g, params, grads)

jit_update=jit(update)

@jit
def accuracy(conv_weights,MLP_params, dataset_imgs, dataset_lbls):

    imgs = conv_apply(conv_weights, dataset_imgs.reshape(-1,28,28,1))
    pred_classes = jnp.argmax(jit_batched_MLP_predict(MLP_params, imgs.reshape(-1,NNin1)), axis=1)

    return jnp.mean(dataset_lbls == pred_classes)
    
jit_accuracy=jit(accuracy)


'''For loop is neccesary to do batch training. Every update iteration needs to run with updated MPL params'''
@jit
def train(conv_weights, imgs, lbls,MLP_params ):
    for n in range(n_training_epochs):  
        for i in range(jnp.shape(imgs)[0]):

            gt_labels = jax.nn.one_hot(lbls[i], 10)
            img_conv = conv_apply(conv_weights, imgs[i].reshape(-1,28,28,1))
            loss, MLP_params = jit_update(MLP_params, img_conv.reshape(-1,NNin1), gt_labels)

    return MLP_params


@jit
def bootstrapp_offspring_MLP(key,conv_weights, batch_affe, labelaffe,test_images,test_lbls):
  
  
    MLP_params = init_MLP([NNin1, NNout1], key)
    MLP_params_trained=train(conv_weights, batch_affe, labelaffe,MLP_params )
    #train(conv_weights, imgs = batch_affe, lbls = labelaffe ,MLP_params = init_MLP )
    
    result=jit_accuracy(conv_weights,MLP_params_trained,test_images,test_lbls)
    return (result)


jit_bootstrapp_offspring_MLP=jit(bootstrapp_offspring_MLP)  


@jit
def vmap_bootstrapp_offspring_MLP(key, conv_weights, batch_affe, labelaffe,test_images,test_lbls):
    return vmap(jit_bootstrapp_offspring_MLP, ( None,None, 0,0,0,0))(key, conv_weights, batch_affe, labelaffe,test_images,test_lbls)
# jax.vmap(fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None
  
jit_vmap_bootstrapp_offspring_MLP=jit(vmap_bootstrapp_offspring_MLP)



def create_offsprings(n_offspr, fath_weights,std_modifier,seed):
    np.random.seed(seed)
    statedic_list=[]
    for i in range(0,n_offspr):
        dicta = [()] * len(father_weights)
        for idx,w in enumerate(father_weights):
            if w:
                w, b = w
                    #print("Weights : {}, Biases : {}".format(w.shape, b.shape))
            
                '''if weight layer only contains 0 and 1, only copy original weight layer, dont add random noise. Purpose of these 0 and 1 layers unclear'''
                if any(w[0].shape==t for t in [(Convu1_in,) ,(Convu2_in,), (Convu3_in,)]):
                    x_w=w
                    x_b=b
                else:
                    seed=np.random.randint(0,100000)
                    key = random.PRNGKey(seed)
                    x_w = w+random.normal(key,shape=w.shape)*std_modifier #tested, random.normal adding different random noise value to every single weight
                    x_b = b+random.normal(key,shape=b.shape)*std_modifier
                dicta[idx]=(x_w,x_b)
        
        statedic_list.append(dicta)
    return statedic_list



def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)



def tree_random_normal_like(rng_key, target,std_modifier):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_multimap(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype)*std_modifier,
        target,
        keys_tree,
    )


def jax_create_offsprings(key,n_offspr,  fath_weights,std_modifier):
    statedic_list=[]
    for i in range(0,n_offspr):
        rng=jax.random.PRNGKey(key+i)
        random_value_tree=tree_random_normal_like(rng,fath_weights,std_modifier)
        son=jax.tree_map(lambda x,y: x+y, fath_weights,random_value_tree)
        statedic_list.append(son)

    return statedic_list


'''softmax for offspring list for approach 2
    checked 11.04 working correctly'''
def softmax_offlist(off_list, acc_list, temp):

    '''Creates softmax/temp list out of accuracy list [0.2,0.3,....,0.8]'''
    def softmax_result(results,temp: float):
        x = [z/temp for z in results]
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    softmax_list=softmax_result(acc_list,temp)
    for i in range(len(off_list)):
        if i==0:
            top_dog=jax.tree_map(lambda x: x*softmax_list[i], off_list[i])
        else:
            general_dog = jax.tree_map(lambda x: x*softmax_list[i], off_list[i])
            top_dog=jax.tree_map(lambda x,y: x+y, top_dog,general_dog)
    return top_dog



def sigma_decay(start, end, n_iter):
    return(end/start)**(1/n_iter)



def KNN_weight_update(result_list_metaepoch, offspring_list):

    offspring_list2=[]
    flat_tw_weight_list=[]

    '''Get weights of top x'''
    acc_list=np.array([x[0] for x in result_list_metaepoch])
    ind_top_n = np.argpartition(acc_list, -Elitist_top_n)[-Elitist_top_n:]
    tw_weight_list=[offspring_list[i] for i in ind_top_n]
    top_acc_list=[acc_list[i] for i in ind_top_n]

    for weight in tw_weight_list:
        affe=jax.flatten_util.ravel_pytree(weight)
        flat_tw_weight_list.append(np.array(affe[0]))
    knn = NearestNeighbors(n_neighbors=KNN_n_neighbors)
    knn.fit(flat_tw_weight_list)
    distance_mat, neighbours_mat = knn.kneighbors(flat_tw_weight_list)
    
    for liste in neighbours_mat:
        weight_list=[tw_weight_list[i] for i in liste]
        acc_list2=[top_acc_list[i] for i in liste]
        new_offspring=softmax_offlist(weight_list,acc_list2,temp)
        offspring_list2.append(new_offspring)
    if use_father:
        offspring_list2.append(tw_weight_list)
    return offspring_list2




def elitist_performer(result_list_metaepoch, offspring_list):
  
    offspring_list2=[]
 
    '''Get weights of top x'''
    acc_list=np.array([x[0] for x in result_list_metaepoch])
    ind_top_n = np.argpartition(acc_list, -Elitist_top_n)[-Elitist_top_n:]
    tw_weight_list=[offspring_list[i] for i in ind_top_n]
    for performer in tw_weight_list:
        new_offsprings=jax_create_offsprings((meta+numpy_seed),n_elitist_offsprings, performer,std_modifier)
        offspring_list2.extend(new_offsprings)
        if use_father:
            offspring_list2.append(performer)
    return offspring_list2



'''Only use one weight update method!'''
'''Initialize Variables'''

'''Use Elitist weight update'''
use_elitist=False #can be combined with Focus_Update
Elitist_top_n=10
n_elitist_offsprings=50

'''Train multiple Google Colabs in parallel'''
use_paralleltraining=False
parallel_path="/content/drive/MyDrive/Colab Notebooks/Jax_MNist/Logs/24.04_parallel/"
      


n_metaepochs=1000
n_offsprings=250
n_samples = 100 #n of training independent training samples for 2nd network - MLP, samples are stratified

n_training_epochs=30 #= how many times, is the same MLP trained with the same data. Reduces dependance on the initialization of MLP weights


batch_size = 25
n_test=1000
n_offsp_epoch=30 #Bootstrapping, delivers more stable results for every offspring. Number of 2nd Networks per Offspringlocal_path

'''keys'''
starting_key=52 #define starting point
MLP_key=369 #seed 
numpy_seed=854 #in create offsprings

use_sigma_decay=True #otherwise using constant sigma from config tab, decreasing sigma for random noise over time
n_decay_epochs=int(n_metaepochs/2)   # over how many metaepochs sigma is decayed
sigma_start=0.01
sigma_goal=0.0000001 #sigma goal after n_metaepochs

use_sigma_randomizer=True #injects increased sigma for more random exploration
std_random_modifier=100
explo_rate=0.1 #how often sigma randomizer is used

'''KNN weight update, disable sigma_decay, start with small sigma'''
use_KNN=False
KNN_n_neighbors=3
KNN_top_n=10
n_KNN_subsprings=50 #number offsprings of every KNN update

use_Softmax=True #weight update method

use_focus=False #only modify weights of one layer of convu. Changing focus every focus_change_every
focus_layer=[0,4,8]
focus_change_every=100

use_pickle=False #load weights
use_best_weights=False
pickle_path="/content/drive/MyDrive/Colab Notebooks/Jax_MNist/Logs/19.04.2022_2parallel/best_weight_0.8865.pkl"
use_father=True
std_modifier=0.05
temp=0.05 #weight for softmax

file_name="JAX_MNist_2"


LEARNING_RATE = 0.001
def UpdateWeights(lr):
    def __UA__(weights, gradients):
        return weights - lr * gradients
    return __UA__

class Encoder(hk.Module):
    def __init__(self):
        super().__init__()
        self.model = hk.Sequential([
            hk.Linear(512), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(64), jax.nn.relu,
            hk.Linear(HIDDEN_SIZE), jax.nn.relu
        ])
        
    def __call__(self, x):
        return self.model(x)
    
class Decoder(hk.Module):
    def __init__(self):
        super().__init__()
        self.model = hk.Sequential([
            hk.Linear(HIDDEN_SIZE), jax.nn.relu,
            hk.Linear(64), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
            hk.Linear(512), jax.nn.relu,
            hk.Linear(28*28)
        ])
        
    def __call__(self, x):
        return self.model(x)
    
class Decoder2(hk.Module):
    def __init__(self):
        super().__init__()
        self.model = hk.Sequential([
            hk.Linear(25*25), jax.nn.relu,
            hk.Linear(28*28)
        ])
        
    def __call__(self, x):
        return self.model(x)
    
class AE(hk.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def __call__(self, x):
        return self.decoder(self.encoder(x))
    
def __hkx_decoder__(x):
    dec = Decoder2()
    return dec(x)

dec = hk.transform(__hkx_decoder__)


def train_encoder(params_conv, key_params_dec, train_input_ds, test_ds, n_epochs):
    dec = hk.transform(__hkx_decoder__)
    
    dummy_x  = jnp.ones([1, 25*25])
    rng_key = jax.random.PRNGKey(100)

    params_dec = dec.init(rng=key_params_dec, x=dummy_x)
    
    @jit
    def __L2Loss__(weights, rng_key, input_data, actual_output):
        preds = dec.apply(weights, rng_key, input_data)
        return jnp.sum(jnp.power(preds - actual_output, 2))
    
    
    for i in range(n_epochs):
        for i_batch in range(train_input_ds.shape[0]):
            batch_x = train_input_ds[i_batch].reshape(-1,28,28,1)
            offspring_output = conv_apply(params_conv, batch_x)
            dec_input = jnp.reshape(offspring_output, [-1, 25*25])

            flat_x = jnp.reshape(batch_x, [-1, 28*28])
            loss, param_dec_grads = value_and_grad(__L2Loss__)(params_dec, key_params_dec, dec_input, flat_x)

            params_dec = jax.tree_map(UpdateWeights(0.00001), params_dec, param_dec_grads) ## Update Params
          
    ## testen     
    
    offspring_output = conv_apply(params_conv,  test_ds.reshape(-1, 28, 28, 1))
    dec_input = jnp.reshape(offspring_output, [-1, 25*25])

    flat_x = jnp.reshape(test_ds.reshape(-1, 28, 28, 1), [-1, 28*28])
    loss = __L2Loss__(params_dec, key_params_dec, dec_input, flat_x)
    return loss

            

    from math import e
#main code
'''Initialize variables'''
f_idx=0
focus_layer=focus_layer*100
rng_MLP=jax.random.PRNGKey(MLP_key)
results_meta=[]
best_performer_ae=10**100
best_performer_class = 0.0
father_key=jax.random.PRNGKey(starting_key)
best_weights=conv_init(father_key, (batch_size,28,28,1))[1]
common_start_acc=0
std_start=std_modifier

'''Start Logging'''
pathandstuff()
#logg_script(file_name, save_path)
log_variables()

summary_writer = SummaryWriter()

for meta in range (n_metaepochs):
    start_meta = time.time()

    '''Sigma Decay'''
    if use_sigma_decay:
        sigma_base=sigma_decay(sigma_start, sigma_goal, n_decay_epochs)
        if meta < n_decay_epochs:
            std_modifier=sigma_start*sigma_base**meta
        else:
            std_modifier=sigma_start*sigma_base**n_decay_epochs

    '''Sigma Randomizer'''
    if use_sigma_randomizer:
        if np.random.uniform(0,1)<explo_rate:
            std_modifier=np.random.uniform(std_modifier/std_random_modifier,std_modifier*std_random_modifier)
            print(f"\trandomized std_modifier: {std_modifier:.4f}")
        else:
            std_modifier=std_start


    '''Starting point'''
    commonweights_loaded=False
    
    
    if use_paralleltraining:
        pass
    """
      '''Check for better weight'''
      best_weights_list=os.listdir(parallel_path)
      if best_weights_list is not None:
        
        for weights in best_weights_list[::-1]:
          if "best_weight" in weights:
            highest_acc=float(weights.split("best_weight_")[1].split(".pkl")[0])
            if highest_acc > common_start_acc:
              commonweights_loaded=True
              with open(parallel_path+f"/best_weight_{highest_acc:.4f}.pkl", "rb") as input_file:
                  father_weights = cPickle.load(input_file)
              logg(f"common weights imported with acc {highest_acc}") 
              common_start_acc=highest_acc
              
              offspring_list=jax_create_offsprings((meta+numpy_seed),n_offsprings, father_weights,std_modifier)
              if use_father:
                offspring_list[0]=father_weights
              best_performer[0]=highest_acc

    """


    if meta ==0 :
        if use_pickle:
            with open(pickle_path, "rb") as input_file:
                father_weights = cPickle.load(input_file)
                print("pickle weights imported") 
            offspring_list=jax_create_offsprings((meta+numpy_seed),n_offsprings, father_weights,std_modifier)
            if use_father:
                offspring_list[0]=father_weights
        
        #wo wird der beste Klassifier ausgewählt?
        else:
            father_weights = conv_init(father_key, (batch_size,28,28,1))
            father_weights = father_weights[1] ## Weights are actually stored in second element of two value tuple
            offspring_list=jax_create_offsprings((meta+numpy_seed),n_offsprings, father_weights,std_modifier)
            if use_father:
                offspring_list[0]=father_weights


    



    '''Weight updates'''      
    if meta >=1 and not commonweights_loaded:

        '''KNN weight update'''
        if use_KNN:
            KNN_offlist=KNN_weight_update(result_list_metaepoch, offspring_list)
            offspring_list=[]
            if use_father:
                offspring_list.append(best_weights)
            for off in KNN_offlist:
                offspring_list.append(off)
                offspring_list.extend(jax_create_offsprings((meta+numpy_seed),n_KNN_subsprings, father_weights,std_modifier))
            

        '''Softmax Update'''
        if use_Softmax:
            grand_father=offspring_list[0]
            father_weights=softmax_offlist(offspring_list,[x[0] for x in result_list_metaepoch],temp)
            offspring_list=jax_create_offsprings((meta+numpy_seed),n_offsprings, father_weights,std_modifier)
            if use_father:
                offspring_list[0]=grand_father
                offspring_list[1]=best_weights
                offspring_list[2]=father_weights
        
        '''Focus Weight Update'''
        if use_focus:
            grand_father=offspring_list[0]
            if meta % focus_change_every ==0:
                f_idx=f_idx+1
                logg(f"Focus change to layer {focus_layer[f_idx]}")
            if use_elitist:
                offspring_list=elitist_performer(result_list_metaepoch, offspring_list)
            else: 
                father_weights=softmax_offlist(offspring_list,[x[0] for x in result_list_metaepoch],temp)
                offspring_list=create_focus_offsprings(n_offsprings, father_weights,std_modifier, focus_layer[f_idx])
            if use_father:
                offspring_list[0]=grand_father
                offspring_list[1]=best_weights
                offspring_list[2]=father_weights

        if use_elitist:
              offspring_list=elitist_performer(result_list_metaepoch, offspring_list)
          
    result_list_metaepoch=[]

    '''same data for every offspring'''
    x_train_unre, x_test_unre, y_train_unre, y_test_unre = train_test_split(x, y, train_size=n_offsp_epoch*n_samples,
                                                          test_size=n_offsp_epoch*n_test,stratify=y,
                                                          random_state=(starting_key+meta))
    
    #hier aufspalten
    
    x_train=x_train_unre.reshape(n_offsp_epoch,int((n_samples/batch_size)),batch_size,28,28,1)
    y_train=y_train_unre.reshape(n_offsp_epoch,int((n_samples/batch_size)),batch_size)
    x_test=x_test_unre.reshape(n_offsp_epoch,n_test,28,28,1)
    y_test=y_test_unre.reshape(n_offsp_epoch,n_test)
    
    print("\tLänge Offspring List:",len(offspring_list))
    #print(f"\tTime overhead: {(time.time()-start_meta):.2f}s")

    use_autoencoder = meta % 20 == 0 and meta > 0 #train autoencoder every 10th epoch
    if use_autoencoder:
        print('its ae time')

    
    for i in range(len(offspring_list)):
        conv_weights=offspring_list[i]
        
        '''
        result_off=jit_vmap_bootstrapp_offspring_MLP(rng_MLP,conv_weights,x_train,y_train,x_test,y_test)
        result_off2=[float(jnp.mean(result_off)),float(jnp.std(result_off))]
        
        '''
        if use_autoencoder:
            result_off = train_encoder(conv_weights, rng_MLP, x_train, x_test, 5)
            result_off = float(result_off)
            result_list_metaepoch.append((float(result_off), 0.0))
            #print(i, result_off)
            summary_writer.add_scalar('training-autoencoder/loss',  result_off, len(offspring_list) * meta + i)


            '''Check for best performer'''
            if result_off<best_performer_ae:
                best_performer_ae=result_off
                best_weights=conv_weights
                common_start_acc=result_off
                with open(save_path+f"best_weight_{result_off:.4f}.pkl", 'wb') as f:
                    pickle.dump(best_weights, f, pickle.HIGHEST_PROTOCOL)
                    f.close()

                logg(f"New best performer mean: {best_performer_ae:.4f}")#, std: {best_performer[1]:.2f}")
        else:
            result_off=jit_vmap_bootstrapp_offspring_MLP(rng_MLP,conv_weights,x_train,y_train,x_test,y_test)
            result_off2=[float(jnp.mean(result_off)),float(jnp.std(result_off))]
            result_list_metaepoch.append(result_off2)
            summary_writer.add_scalar('training-classifier/accuracy-mean',  
                                    np.array(result_off2[0]), len(offspring_list) * meta + i)
            summary_writer.add_scalar('training-classifier/accuracy-std',  
                                    np.array(result_off2[1]), len(offspring_list) * meta + i)


            '''Check for best performer'''
            if result_off2[0] > best_performer_class:
                best_performer_class = result_off2[0]
                best_weights = conv_weights
                common_start_acc = result_off2[0]
                with open(save_path+f"best_weight_{result_off2[0]:.4f}.pkl", 'wb') as f:
                    pickle.dump(best_weights, f, pickle.HIGHEST_PROTOCOL)
                f.close()
                        
    

    #logg("\tMetaepoch mean: {:.4f}, std: {:.2f}".format(np.mean(np.array([x[0] for x in result_list_metaepoch])),np.std(np.array([x[0] for x in result_list_metaepoch]))))
    #logg("\tMetaepoch max performer: {:.4f}, min performer: {:.4f}".format(np.max(np.array([x[0] for x in result_list_metaepoch])),np.min(np.array([x[0] for x in result_list_metaepoch]))))
    #logg("\tTime per metaepoch:{:.1f}s\n".format(time.time() - start_meta))
    results_meta.append(np.mean(np.array(result_list_metaepoch), axis=0))

    task_name = "autoencoder" if use_autoencoder else "classifier"

    def __is_worthy_data__(y):
        return not (isnan(y) or isinf(y))
    max_meta = max([x[0] for x in result_list_metaepoch if __is_worthy_data__(x[0])])
    min_meta = min([x[0] for x in result_list_metaepoch if __is_worthy_data__(x[0])])
    mean_meta = mean([x[0] for x in result_list_metaepoch if __is_worthy_data__(x[0])])
    std_meta = stdev([x[0] for x in result_list_metaepoch if __is_worthy_data__(x[0])])
    median_meta = median([x[0] for x in result_list_metaepoch if __is_worthy_data__(x[0])])

    summary_writer.add_scalar(f"meta-{task_name}/max", max_meta, meta)
    summary_writer.add_scalar(f"meta-{task_name}/min", min_meta, meta)
    summary_writer.add_scalar(f"meta-{task_name}/mean", mean_meta, meta)
    summary_writer.add_scalar(f"meta-{task_name}/std", std_meta, meta)
    summary_writer.add_scalar(f"meta-{task_name}/median", median_meta, meta)



    summary_writer.add_scalar(f"meta-all/max", max_meta, meta)
    summary_writer.add_scalar(f"meta-all/min", min_meta, meta)
    summary_writer.add_scalar(f"meta-all/mean", mean_meta, meta)
    summary_writer.add_scalar(f"meta-all/std", std_meta, meta)
    summary_writer.add_scalar(f"meta-all/median", median_meta, meta)


    print("=" * 20)
    print(max_meta, min_meta, mean_meta, std_meta, median_meta)

    if use_autoencoder:
        result_list_metaepoch2 = list()
        for result_one in result_list_metaepoch:
            if isnan(result_one[0]) or isinf(result_one[0]):
                result_list_metaepoch2.append((-10.0, result_one[1]))
            else:
                result_list_metaepoch2.append(
                    ( (1 - result_one[0]/max_meta) * 100, result_one[1])
                    )
        result_list_metaepoch = result_list_metaepoch2

summary_writer.close()
