from matplotlib import rcParams
from joblib import Parallel, delayed
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image


import tensorflow.keras as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Activation, AveragePooling2D, MaxPooling2D, Dropout, BatchNormalization
from scipy import stats
# from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Nadam

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os, random
import sys
import pandas as pd
import glob

gt_path = 'gt/'
gt_labels_path = gt_path + 'label_images/'
gt_labels_path_npy = gt_labels_path + 'npy/'
images_path = 'images/'
images_path_npy = images_path + 'npy/'




# Retrieve colors from csv
class_colors_df = pd.read_csv(gt_path + 'class_dict.csv')
class_colors_df = class_colors_df[0:-1] # Remove last class, because no pixels are classified as such.
class_names = class_colors_df.loc[:, 'name'].values.tolist()
class_colors = class_colors_df.loc[:, [False, True, True, True]].values
nclasses = class_colors.shape[0]


# Generate numpy file (.npy) with all images and save to 'npy' folder.

def generate_training_var(normalize=False):
  all_imgs_path = glob.glob(images_path + '*.jpg')
  all_imgs_path = [add.split('/')[-1] for add in all_imgs_path]
  imgs_numpy = list()
  
  if normalize:
    if normalize:
      for img_path in all_imgs_path:
        img = np.array( Image.open(images_path + img_path).convert('RGB') ).astype( np.uint8 )
        img_float = img.astype(np.float32)
        desc = stats.describe(img_float, axis = None)
        img_float = (img_float - desc.mean )/desc.variance
        imgs_numpy.append( img_float )
  else:
    for img_path in all_imgs_path:
      img = np.array( Image.open(images_path + img_path).convert('RGB') ).astype( np.uint8 )
      imgs_numpy.append( img )
  imgs_numpy = np.array( imgs_numpy )
  img_names = [allimg.split('.')[0] for allimg in all_imgs_path]
  return imgs_numpy, img_names

# Convert ground truth to one-hot representation and save to 'npy' folder.
def generate_onehot_gt():
  #First convert to numpy
  all_gts_path = glob.glob(gt_labels_path + '*.png')
  all_gts_path = [add.split('/')[-1] for add in all_gts_path]
  Gts_numpy = list()
  for gt in all_gts_path:
    img = np.array( Image.open(gt_labels_path + gt).convert('RGB') ).astype( np.uint8 )
    # img = np.swapaxes( np.swapaxes(img, 1, 2), 0, 1)  # PyTorch shape (nchannels, npix_vert, npix_horiz)
    Gts_numpy.append( img )

  Gts_numpy = np.array(Gts_numpy) # Shape = (nimages, nchannels, npix_vert, npix_horiz )
  # Y_full = np.zeros( ( Gts_numpy.shape[0], nclasses, Gts_numpy.shape[2], Gts_numpy.shape[3] ), dtype = np.uint8 ) # Shape = (nimages, nclasses, npix_vert, npix_horiz ). We substitute the 3 RGB channels for a one-hot vector of dimensions 'nclasses'.
  Y_full = np.zeros( ( Gts_numpy.shape[0], Gts_numpy.shape[1], Gts_numpy.shape[2], nclasses ), dtype = np.uint8 ) # Shape = (nimages, nclasses, npix_vert, npix_horiz ). We substitute the 3 RGB channels for a one-hot vector of dimensions 'nclasses'.
  Pixels_per_class = np.zeros(nclasses)


  for gt_idx, gt_img in enumerate(Gts_numpy):
    for class_idx, class_color in enumerate(class_colors):
      coord_pixels_of_a_class = np.logical_and( np.logical_and( ( gt_img[:, :, 0] == class_color[0] ) , gt_img[:, :, 1] == class_color[1] ),  gt_img[:, :, 2] == class_color[2] ) # Retrieve pixel coordinates where class 'class_color' is present.
      Y_full[gt_idx, :, :, class_idx][coord_pixels_of_a_class] = 1
      Pixels_per_class[class_idx] += np.sum(coord_pixels_of_a_class)
  return Y_full, Pixels_per_class


# Exploratory Analysis

# Retorna uma lista com os objetos presentes em uma imagem
def parse_XML(xml_file, df_cols): 
    """
    The first element of df_cols is supposed to be the identifier 
    variable, which is an attribute of each node element in the 
    XML data; other features will be parsed from the text content 
    of each sub-element. 
    """
    
    xtree = et.parse(xml_file, et.XMLParser(encoding='utf-8'))
    xroot = xtree.getroot()
    object_list = []
    #rows = []

    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        #rows.append({df_cols[i]: res[i] for i, _ in enumerate(df_cols)})
        object_list.append(res[1])
    
    #out_df = pd.DataFrame(rows, columns=df_cols)
        
    return object_list #out_df

# Retorna uma lista com elementos únicos sem valor None e layer
def get_unique(lista):
    unique_list = []
    unique_set = set(lista)
    for element in unique_set:
        if element is None or element=="layer":
          continue
        unique_list.append(element)
    return unique_list

# Include color info in DF
def get_color_code(df):
  class_colors='/content/SemanticDroneDataset/gt/class_dict_norm.csv'
  colorcode = pd.read_csv(class_colors, names=['name','r','g','b'] )
  colorcode['color'] = list(zip(colorcode.r, colorcode.g,colorcode.b))
  colorcode = colorcode[colorcode.name != 'unlabeled']
  colorcode = colorcode[colorcode.name != 'conflicting']
  colorcode = colorcode[colorcode.name != 'name']

  colorcode.sort_values('name', inplace=True)
  df.sort_values('Name', inplace=True)
  df['color'] = colorcode['color']
  return df #color 

def activation_statistics(X, activation_model, act_index):
  X = np.expand_dims(X, axis=0)
  # X = preprocess_input(X)
  activations = activation_model.predict(X)
  activation = activations[act_index][0]
  
  # Statistical description of flattened activation
  flat_act = np.matrix.flatten(activation)
  statistics = stats.describe(flat_act)
  mean = statistics.mean
  sdev = statistics.variance**(1/2)
  skewness = statistics.skewness
  kurtosis = statistics.kurtosis

  # Pixel-wise statistical description (forms an image of shape (activation.shape[1], activation.shape[2] ) )
  attributes_pixw = stats.describe(activation, axis = -1)
  attributes = np.array([mean, sdev, skewness, kurtosis])
  return attributes, attributes_pixw


def data_statistics_table(X_full, img_names, act_index, save_name = None, model = None):
  if save_name is None:
    if act_index != -1:
      save_name = "table_attributes_actv_" + str(act_index)
    else:
      save_name = "table_attributes_actv_last"
  SAVE = 1
  table_contents = []
  pixelwise_stats = []
  num_images = X_full.shape[0]
  pbar = progressbar.ProgressBar(max_value=num_images)
  if model is None: 
    model = Xception(weights='imagenet', input_shape = (400, 600, 3), include_top = False ) # Xception without FullyConnected layer (last).
  layer_outputs = [layer.output for layer in model.layers]
  activation_model = Model(inputs=model.input, outputs=layer_outputs)
  for i, X in enumerate(X_full):
    attributes, stats_pixw = activation_statistics(X, activation_model, act_index)
    table_contents.append(attributes)
    pixelwise_stats.append(stats_pixw)
    pbar.update(i+1)
  table_contents = np.array( table_contents )
  pixelwise_stats = np.array( pixelwise_stats, dtype=object )
  table_df = pd.DataFrame(table_contents, columns = atributes_names)
  table_df.insert(0, "img_name", img_names, True)
  table_df = table_df.sort_values("img_name")
  savepath = 'tables/' + save_name + '.csv'
  table_df.to_csv(savepath)
  return table_df, table_contents, pixelwise_stats

def plot_statistics(table_name):
  figname = 'pairgrid_' + table_name
  table_df = pd.read_csv('tables/' + table_name + '.csv')
  columns_to_ignore = table_df.columns[[0, 1]]
  table_df.drop(columns_to_ignore, inplace=True, axis=1)
  scatter = sns.PairGrid(table_df)
  scatter.map_diag(sns.histplot, kde=True, hue = None)
  scatter.map_offdiag(sns.scatterplot)
  for ax in scatter.axes.flat:
    # labelleft refers to yticklabels on the left side of each subplot
    # ax.tick_params(axis='y', labelleft=True) # method 1
    ax.tick_params(axis='x', labelbottom=True)
  plt.show
  plt.savefig( analise_dir + figname + ".eps", format="eps")

def plot_correlation_heatmap(table_name,figname = None):
  if figname is None:
    figname = 'fig_' + str(table_name)
  #table_df = pd.read_csv('tables/' + table_name + '.csv')
  table_df = pd.read_csv('tables/' + table_name + '.csv')
  columns_to_ignore = table_df.columns[[0, 1]]
  table_df.drop(columns_to_ignore, inplace=True, axis=1)
  corr_table = table_df.corr() # Pearson's correlation coefficient.
  # mask_tri = 1 - np.tri(corr_table.shape[0])
  f, ax = plt.subplots(figsize=(7, 7))
  ax = sns.heatmap(corr_table, center = 0, vmin=-1, vmax=1, annot=True, xticklabels=False, cmap=cm.twilight_shifted)  #TODO novos parametros
  ax.set_yticklabels(ax.get_yticklabels(), rotation=30, ha='right') 
  ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')  #Se quiser que apareça o texto em x, seta True ali em cima
  sns.set(font_scale=2)

  # Matplotlib general parameters   (copy and paste in other plots)    # TODO verificar se usa os mesmos parâmetros do histograma
  matplotlib.rcParams["font.family"] = "Dejavu Serif"
  # plt.style.use(['seaborn-colorblind']) 
  plt.rcParams['font.size'] = 16
  plt.rcParams['axes.linewidth'] = 2
  plt.style.use(['seaborn-colorblind'])
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(True)

  #Visualize and save
  plt.show
# if save_img:
  plt.savefig( analise_dir + figname + ".eps", format="eps", bbox_inches='tight') # TODO verificar se o comando tá dando certo aqui


def display_activation(X, act_index = -1, model = None):
  if model is None: 
    model = Xception(weights='imagenet', input_shape = (400, 600, 3), include_top = False ) # Xception without FullyConnected layer (last).
  X = np.expand_dims(X, axis=0)
  X = preprocess_input(X)
  layer_outputs = [layer.output for layer in model.layers]
  activation_model = Model(inputs=model.input, outputs=layer_outputs)
  print(activation_model.summary())
  activations = activation_model.predict(X)
  activation = activations[act_index]
  number_of_filters = len( activation[0, 0, 0] )
  if number_of_filters <= 256:
    col_size = 4
    row_size = int( np.ceil(number_of_filters/col_size) )
  else:
    number_of_filters = 256
    row_size = 64
    col_size = 4
  zoom = 4
  figshape = (col_size*(zoom*3/2), row_size*zoom) # 3 by 2 proportion
  fig, ax = plt.subplots(row_size, col_size, figsize = figshape)
  activation_index=0
  for row in range(row_size):
    for col in range(col_size):
      ax[row][col].set_axis_off()
      ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
      activation_index += 1




#Preprocessing


# Por favor não usar 'image' como variável, porque é o nome de uma classe do Keras. Pode usar 'Image. #ok!
def show_transformation(img_name,title=None,save=False):
  Image, _, Gt_RGB = load_img_and_augment(img_name)
  fig, ax = plt.subplots(nrows=1, ncols=1)
  ax.imshow(Image.astype(np.uint8))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  if title is not None:
    ax.set_title(title, pad = 20, fontsize=36)
  plt.show
  if save:
    plt.savefig( analise_dir + title + ".eps", format="eps") # TODO verificar se o comando tá dando certo aqui

  #plt.savefig(f"{analise_dir}/classe-hist-img.eps")#, format="eps")


# Essas variáveis são definidas lá em cima já.
# working_dir = '/content/SemanticDroneDataset/'
# dataset_dir = base_dir+'SemanticDroneDataset/' 
# backup_dir = base_dir+'SemanticDroneDataset_backup/'
# analise_dir = working_dir+'analise/' 


def show_image(img_name):
  img_file = np.asarray( image.load_img(images_path + img_name + '.jpg'), dtype=np.uint8 )
  fig, ax = plt.subplots(nrows=1, ncols=1)
  ax.imshow(img_file)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)