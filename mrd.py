
import sys
import argparse
import numpy as np
import pandas as pd
import os
import shutil
import logging
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from utilities import *
from myModules import *
args = sys.argv
"""
embファイル、componentファイル、labelファイルから、modelが欲しい形に翻訳する
INPUT: embファイル、componentファイル、labelファイル(2つ)
"""
def specify_which(source_labels, val_labels, target_labels, disc_data):
    """disc_dataの中身を解析し、disc_dataのどっちがsourceでどっちがテストか"""
    # TODO: refactoring
    if source_labels.shape[0] == target_labels.shape[0]:
        # _logger.warning("target and source size is the same. Check to see source and target is separated as you want.")
        print("target and source size is the same. Check to see source and target is separated as you want.")
    label0_num = disc_data[disc_data.component_label == 0].shape[0]
    label1_num = disc_data[disc_data.component_label == 1].shape[0]
    label2_num = disc_data[disc_data.component_label == 2].shape[0]
    label_nums = [0, 1, 2]

    if label0_num == source_labels.shape[0]:
        source_label = 0
    elif label1_num == source_labels.shape[0]:
        source_label = 1
    elif label2_num == source_labels.shape[0]:
        source_label = 2
    else:
        raise ValueError("source label cannot be specified")
    label_nums.remove(source_label)

    if label0_num == val_labels.shape[0]:
        val_label = 0
    elif label1_num == val_labels.shape[0]:
        val_label = 1
    elif label2_num == val_labels.shape[0]:
        val_label = 2
    else:
        raise ValueError("Val label cannot be specified")
    label_nums.remove(val_label)

    target_label = label_nums[0]
    return source_label, val_label, target_label

def specify_which_r2(source_labels, val_labels, target_labels, disc_data):
    """disc_dataの中身を解析し、disc_dataのどっちがsourceでどっちがテストか"""
    # TODO: refactoring
    return 0, 2, 1



def set_parser_ts():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", default="graph/labels-usa-airports.txt",
                        help="source labels file")
    parser.add_argument("--target", default=None,
                        help="target labels file")
    parser.add_argument("--validation", default=None,
                        help="Validation labels file")
    parser.add_argument("emb", default="emb/usa_europe.emb",
                        help="Emb file")
    parser.add_argument("component", default="graph/usa_europe.component",
                        help="Emb file")
    args = parser.parse_args()
    return args


def prepare_discriminator(disc_data: pd.DataFrame, source_label: int, val_label: int, target_label: int, train_discriminator: str):
    source_disc = disc_data[disc_data.component_label == source_label].sort_values(by="node")
    val_disc = disc_data[disc_data.component_label == val_label].sort_values(by="node")
    target_disc = disc_data[disc_data.component_label == target_label].sort_values(by="node")

    source_disc.component_label = 0
    target_disc.component_label = 1
    if train_discriminator == 'd2':
        val_disc.component_label = 1
    elif train_discriminator == 'd3':
        val_disc.component_label = 2
    else:
        val_disc.component_label = 1
        
    disc_data = pd.concat([target_disc, val_disc, source_disc], axis=0)   # [1, 1, ,1, ,1 , ....., 0, 0, 0] euなら1, usaなら0
    disc_data = disc_data.component_label.values.astype(np.float32)
    return disc_data


def prepare_features(features, source_label, val_label, target_label):
    source_features = features[features.component_label == source_label].sort_values(by=0).drop(["component_label", "node", 0], axis=1).values.astype(np.float32)
    val_features = features[features.component_label == val_label].sort_values(by=0).drop(["component_label", "node", 0], axis=1).values.astype(np.float32)
    target_features = features[features.component_label == target_label].sort_values(by=0).drop(["component_label", "node", 0], axis=1).values.astype(np.float32)
    # import pdb; pdb.set_trace()
    if val_features.shape[0] != 0:
        target_features = np.concatenate([val_features, target_features], axis=0)
    return source_features, target_features


# def prepare_labels(source_labels, val_labels, target_labels):
#     source_labels = source_labels.sort_values(by="node").label.values.astype(np.float32)
#     val_labels = val_labels.sort_values(by="node").label.values.astype(np.float32)
#     target_labels = target_labels.sort_values(by="node").label.values.astype(np.float32)
#     return source_labels, val_labels, target_labels

def prepare_labels(source_labels, val_labels, target_labels):
    source_labels = source_labels.sort_values(by="node")[list(source_labels.columns)[1:len(list(source_labels.columns))-TRAIN_TRIMNO]].values.astype(np.float32)
    val_labels = val_labels.sort_values(by="node")[list(val_labels.columns)[1:len(list(val_labels.columns))-TRAIN_TRIMNO]].values.astype(np.float32)
    target_labels = target_labels.sort_values(by="node")[list(target_labels.columns)[1:len(list(target_labels.columns))-TRAIN_TRIMNO]].values.astype(np.float32)

    # source_labels = source_labels.sort_values(by="node")[list(source_labels.columns)[1:len(list(source_labels.columns))]].values.astype(np.float32)
    # val_labels = val_labels.sort_values(by="node")[list(val_labels.columns)[1:len(list(val_labels.columns))]].values.astype(np.float32)
    # target_labels = target_labels.sort_values(by="node")[list(target_labels.columns)[1:len(list(target_labels.columns))]].values.astype(np.float32)
    return source_labels, val_labels, target_labels

def exec_translate(source_label_file, emb_file, component_file, target_label_file, validation_label_file, train_discriminator): # para1:source_label, p2:emb, p3:component, p4:target_label, p5:validation_label
    # 必要なファイルの読み込み
    features = pd.read_csv(emb_file, sep=" ", skiprows=1, header=None)
    disc_data = pd.read_csv(component_file, sep=" ")  # 0->usa, 1->eu
    source_labels = pd.read_csv(source_label_file, sep=" ")
    if target_label_file is not None:
        target_labels = pd.read_csv(target_label_file, sep=" ")
    else:
        target_labels = pd.DataFrame(columns=["label", "node"])

    if validation_label_file is not None:
        val_labels = pd.read_csv(validation_label_file, sep=" ")
    else:
        val_labels = pd.DataFrame(columns=["label", "node"])

    # _logger.info("source_size: {}, val_size: {}, target_size: {}".format(
    #             source_labels.shape[0], val_labels.shape[0], target_labels.shape[0]))
    print ("0 source_size: {}, val_size: {}, target_size: {}, feature_size: {}, disc_size{}".format(
                source_labels.shape, val_labels.shape, target_labels.shape, features.shape, disc_data.shape))
    # sourceとtargetのラベルの設定
    # source_label, val_label, target_label = specify_which(source_labels, val_labels, target_labels, disc_data)
    source_label, val_label, target_label = 0,2,1
    features = pd.merge(left=features, right=disc_data, left_on=0, right_on="node", how="inner")
    print ("1 source_size: {}, val_size: {}, target_size: {}".format(
                source_label, val_label, target_label))

    # それぞれinputの形へ
    source_features, target_features = prepare_features(features, source_label, val_label, target_label)
    # print ("debug sf={}, tf={}".format(source_features.shape, target_features.shape))
    disc_data = prepare_discriminator(disc_data, source_label, val_label, target_label, train_discriminator)
    source_labels, val_labels, target_labels = prepare_labels(source_labels, val_labels, target_labels)

    print ("2 source_size: {}, val_size: {}, target_size: {}, feature_size: {}".format(
                source_labels.shape, val_labels.shape, target_labels.shape, features.shape))
    # npyファイルで吐き出す
    if os.path.exists(HOME+"dump"):
        import shutil
        shutil.rmtree(HOME+"dump")
    os.makedirs(HOME+"dump")

    np.save(HOME+"dump/source_features.npy", source_features)
    np.save(HOME+"dump/target_features.npy", target_features)
    np.save(HOME+"dump/source_labels.npy", source_labels)
    np.save(HOME+"dump/disc_data.npy", disc_data)
    if target_label_file is not None:
        np.save(HOME+"dump/target_labels.npy", target_labels)
    if validation_label_file is not None:
        np.save(HOME+"dump/val_labels.npy", val_labels)
    # _logger.info("successfully saved numpy file!")
    print ("successfully saved numpy file!")


def translator(source_label_file, emb_file, component_file, target_label_file, validation_label_file, train_discriminator='d2'): # para1:source_label, p2:emb, p3:component, p4:target_label, p5:validation_label
    exec_translate(source_label_file, emb_file, component_file, target_label_file, validation_label_file, train_discriminator) # para1:source_label, p2:emb, p3:component, p4:target_label, p5:validation_label
# device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


def set_parser_tr():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_target_label", action="store_true",
                        help="True if target labels are available for calculating target accuracy")

    parser.add_argument("--is_val_label", action="store_true",
                        help="True if validation labels are available for calculating validation accuracy")

    parser.add_argument("--param_dir", default=None,
                        help="preserve features and predicted labels")

    parser.add_argument("--epoch", default=3000, type=int,
                        help="How many times models iterate")

    parser.add_argument("--dr_rate", default=0.5, type=float,
                        help="Drop out rate should be set from 0 to 1")

    parser.add_argument("--lambda_", default=10, type=float,
                        help="Balance coefficient")

    parser.add_argument("--wd", default=1e-4, type=float,
                        help="weight decay rate")

    parser.add_argument("--r_lr", default=1e-4, type=float,
                        help="learning rate for Role-Model")

    parser.add_argument("--d_lr", default=1e-4, type=float,
                        help="learning rate for Discriminator")

    parser.add_argument("--suffix", default="adv", type=str, 
                        help="Suffix for this implementation, used for saving params, logging")

    parser.add_argument("--id", default=5, type=int, 
                        help="Id for this test set, used for identifying test progress")

    parser.add_argument("--randomseed", default=19890904, type=int, 
                        help="Random seed for torch, to guarantee reproducibility")

    parser.add_argument("--is_visualize", action="store_true",
                        help="True if output visualization results")

    args = parser.parse_args()
    return args

def save_params(param_dir, suffix, after_source_X, after_target_X, val_outputs, target_outputs, source_outputs, val_threshold, target_threshold, source_threshold):
    # if not os.path.exists(param_dir):
    #     os.mkdir(param_dir)
    if not os.path.exists(HOME+param_dir):
        os.mkdir(HOME+param_dir)
    else:
        shutil.rmtree(HOME+param_dir)
        os.mkdir(HOME+param_dir)

    np.save(HOME+param_dir+"/source_feature_{}.npy".format(suffix), after_source_X)
    np.save(HOME+param_dir+"/target_feature_{}.npy".format(suffix), after_target_X)
    np.save(HOME+param_dir+"/valida_outputs_{}.npy".format(suffix), val_outputs)
    np.save(HOME+param_dir+"/target_outputs_{}.npy".format(suffix), target_outputs)
    np.save(HOME+param_dir+"/source_outputs_{}.npy".format(suffix), source_outputs)
    np.save(HOME+param_dir+"/valida_threshold_{}.npy".format(suffix), val_threshold)
    np.save(HOME+param_dir+"/target_threshold_{}.npy".format(suffix), target_threshold)
    np.save(HOME+param_dir+"/source_threshold_{}.npy".format(suffix), source_threshold)


def train(D, D_criterion, D_labels, D_optimizer, R, R_criterion, y_source, R_optimizer, lambda_, X_source):
    # Discriminator optimization
    D.train()
    detached_weight = R.embed.weight.detach()
    D_outputs = D(detached_weight)

    D_loss = lambda_ * D_criterion(D_outputs, D_labels)
    print (D_loss) #debug
    D_optimizer.zero_grad()
    D_loss.backward(retain_graph=True)
    D_optimizer.step()

    # Role-model optimization
    R.train()
    R_outputs = R(X_source)

    move_weight = R.embed.weight
    D_outputs = D(move_weight)
    # print (D_outputs.shape, D_labels.shape,y_source.shape, R_outputs.shape) # debug
    R_loss = R_criterion(R_outputs, y_source.float()) \
                - lambda_ * D_criterion(D_outputs, D_labels)
    print (R_loss) #debug
    R_optimizer.zero_grad()
    R_loss.backward()
    R_optimizer.step()

    return R_outputs, D_outputs

def main(is_target_label, randomseed, id, is_visualize, is_val_label, save_para=False):# para1:is_target_label 2:randomseed 3:id 4:is_visualize 5:is_validation

  epochs = [500, 1000, 5000]
  dr_rates = [0.25, 0.5]
  lambda_s = [0.01, 0.1, 1, 10, 100]
  weight_decays = [1e-4]
  task_lrs = [1e-2, 1e-3, 1e-4, 1e-5]
  disc_lrs = [1e-2, 1e-3, 1e-4, 1e-5]

#   epochs = [500]
#   dr_rates = [ 0.25, 0.5]
#   lambda_s = [1]
#   weight_decays = [1e-4]
#   task_lrs = [1e-2]
#   disc_lrs = [1e-2]

  # set random seed
  torch.manual_seed(randomseed)

  # Read latent representations
  source_X = np.load(HOME+"dump/source_features.npy")
  source_labels = np.load(HOME+"dump/source_labels.npy")

  target_X = np.load(HOME+"dump/target_features.npy")
  D_labels = np.load(HOME+"dump/disc_data.npy")

  # MLSMOTE
  if TRAIN_IMB == 'MLSMOTE':
    print ('MLSMOTE bef',source_X.shape, source_labels.shape, D_labels.shape)
    bef_size_source = source_X.shape[0]
    X_df = pd.DataFrame(source_X)
    y_df = pd.DataFrame(source_labels)
    X_df, y_df = MLSMOTE(X_df, y_df, ratio='auto', neigh=5, thres=0, ql=[0,1], seed=123)
    source_X = X_df.values.astype(np.float32)
    source_labels = y_df.values.astype(np.float32)
    D_labels = np.append(D_labels, np.zeros(source_X.shape[0] - bef_size_source))
    print ('MLSMOTE aft',source_X.shape, source_labels.shape, D_labels.shape)

  merge_X = np.concatenate([target_X, source_X], axis=0)

  # Read labels
  if is_target_label:
    target_labels = np.load(HOME+"dump/target_labels.npy")
    y_target = torch.tensor(target_labels, requires_grad=False).to(device)
  if is_val_label:
    val_labels = np.load(HOME+"dump/val_labels.npy")
    y_val = torch.tensor(val_labels, requires_grad=False).to(device)
    val_size = val_labels.shape[0]

  # Read source or target label
  D_labels = torch.tensor(D_labels, requires_grad=False).to(device)
  y_source = torch.tensor(source_labels, requires_grad=False).to(device)

  target_size = target_X.shape[0]

  # input of models
  X = torch.arange(merge_X.shape[0])
  X_source = X[target_size:].to(device)
  X_target = X[:target_size].to(device)
  
  # index for multi discriminator's input
  if is_target_label and is_val_label and TRAIN_MULTIDISCRIMINATOR:
    disc_input_target = list(range(val_size,target_size)) + list(range(target_size,merge_X.shape[0]))
    disc_input_valida = list(range(val_size)) + list(range(target_size,merge_X.shape[0]))

  max_val_accuracies={}
  startTimeStr=str(datetime.datetime.now())

  counter = 0
  if save_para:
    os.makedirs(HOME+'/para'+str(randomseed))
  max_val_accuracy = 0
  max_val_counter = 0
  source_feature_ = None
  target_feature_ = None
  valida_outputs_ = None
  target_outputs_ = None
  source_outputs_ = None
  valida_threshold_ = None
  target_threshold_ = None
  source_threshold_ = None
  r_list = []

  for epoch in epochs:
    for lambda_ in lambda_s:
      for dr_rate in dr_rates:
        for weight_decay in weight_decays:
          for task_lr in task_lrs:
            for disc_lr in disc_lrs:
              # for commit-ml
              hparams = {
                  "epochs": epoch,
                  "lambda_": lambda_,
                  "dr_rate": dr_rate,
                  "weight_decay": weight_decay,
                  "task_lr": task_lr,
                  "disc_lr": disc_lr
              }
              
              # torch.manual_seed(randomseed)
              hparam_suffix = "e{}_dr{}_wd_{}_tlr{}_dlr{}_lamda{}".format(
                          hparams["epochs"], hparams["dr_rate"], hparams["weight_decay"],
                          hparams["task_lr"], hparams["disc_lr"], hparams["lambda_"])
              # model definition
            #   R = RoleModel(init_features=merge_X, dr_rate=hparams["dr_rate"], class_num=len(y_source.unique())).to(device)

              if TRAIN_LOSS == 'ibpmll':
                R = RoleModel_ibpmll(init_features=merge_X, dr_rate=hparams["dr_rate"], class_num=len(y_source[0])).to(device)
              elif TRAIN_LOSS == 'bpmll':
                R = RoleModel(init_features=merge_X, dr_rate=hparams["dr_rate"], class_num=len(y_source[0])).to(device)  
              elif TRAIN_LOSS == 'bce':
                R = RoleModel(init_features=merge_X, dr_rate=hparams["dr_rate"], class_num=len(y_source[0])).to(device)  
              else:
                R = RoleModel(init_features=merge_X, dr_rate=hparams["dr_rate"], class_num=len(y_source[0])).to(device)  
            
              if is_target_label and is_val_label and TRAIN_MULTIDISCRIMINATOR:
                Dt = Discriminator(hparams["dr_rate"], emb_size=merge_X.shape[1], discriminatorTpye=TRAIN_DISCRIMINATOR).to(device)
                Dv = Discriminator(hparams["dr_rate"], emb_size=merge_X.shape[1], discriminatorTpye=TRAIN_DISCRIMINATOR).to(device)
                # loss and optimizer            
                if TRAIN_DISCRIMINATOR == 'd3':
                  Dt_criterion = nn.CrossEntropyLoss()
                  Dv_criterion = nn.CrossEntropyLoss()
                  Dt_labels = D_labels[disc_input_target].long()
                  Dv_labels = D_labels[disc_input_valida].long()
                else:
                  Dt_criterion = nn.BCELoss()
                  Dv_criterion = nn.BCELoss()
                  Dt_labels = D_labels[disc_input_target].float()
                  Dv_labels = D_labels[disc_input_valida].float()
                Dt_optimizer = torch.optim.Adam(Dt.parameters(), lr=hparams["disc_lr"],  weight_decay=hparams["weight_decay"])
                Dv_optimizer = torch.optim.Adam(Dv.parameters(), lr=hparams["disc_lr"],  weight_decay=hparams["weight_decay"])
              else:
                D = Discriminator(hparams["dr_rate"], emb_size=merge_X.shape[1], discriminatorTpye=TRAIN_DISCRIMINATOR).to(device)
                # loss and optimizer
                if TRAIN_DISCRIMINATOR == 'd3':
                  D_criterion = nn.CrossEntropyLoss()
                  D_labels = D_labels.long()
                else:
                  D_criterion = nn.BCELoss()
                  D_labels = D_labels.float()
                D_optimizer = torch.optim.Adam(D.parameters(), lr=hparams["disc_lr"],  weight_decay=hparams["weight_decay"])
                
              if TRAIN_LOSS == 'ibpmll':
                R_criterion = I_BPMLLLoss()
              elif TRAIN_LOSS == 'bpmll':
                R_criterion = BPMLLLoss()
              elif TRAIN_LOSS == 'bce':
                R_criterion = nn.BCELoss()
              else:
                R_criterion = nn.BCELoss()
              
              R_optimizer = torch.optim.Adam(R.parameters(), lr=hparams["task_lr"], weight_decay=hparams["weight_decay"])
              for e in range(hparams["epochs"]):
                # train epoch
                # Discriminator optimization
                if is_target_label and is_val_label and TRAIN_MULTIDISCRIMINATOR:
                  Dt.train()
                  Dv.train()
                  detached_weight = R.embed.weight.detach()
                  Dt_outputs = Dt(detached_weight[disc_input_target])
                  Dv_outputs = Dv(detached_weight[disc_input_valida])
                  Dt_loss =  Dt_criterion(torch.squeeze(Dt_outputs), D_labels[disc_input_target])
                  Dv_loss =  Dv_criterion(torch.squeeze(Dv_outputs), D_labels[disc_input_valida])
                  Dt_optimizer.zero_grad()
                  Dv_optimizer.zero_grad()
                  Dt_loss.backward(retain_graph=True)
                  Dv_loss.backward(retain_graph=True)
                  Dt_optimizer.step()
                  Dv_optimizer.step()
                  # Role-model optimization
                  R.train()
                  R_outputs = R(X_source)
                  move_weight = R.embed.weight
                  Dt_outputs = Dt(move_weight[disc_input_target])
                  Dv_outputs = Dv(move_weight[disc_input_valida])
                  if TRAIN_DISCRIMINATOR_LOSS == 'half':
                    R_loss = (R_criterion(R_outputs, y_source.float())
                            - hparams["lambda_"] * Dt_criterion(torch.squeeze(Dt_outputs), D_labels[disc_input_target])
                            - hparams["lambda_"] * Dv_criterion(torch.squeeze(Dv_outputs), D_labels[disc_input_valida]))
                  elif TRAIN_DISCRIMINATOR_LOSS == 'nomalized':
                    # R_loss = (R_criterion(R_outputs, y_source.float())
                    #         - hparams["lambda_"] * Dt_criterion(torch.squeeze(Dt_outputs), D_labels[disc_input_target]) * 2 * (target_size - val_size) / target_size 
                    #         - hparams["lambda_"] * Dv_criterion(torch.squeeze(Dv_outputs), D_labels[disc_input_valida]) * 2 * (val_size) / target_size )
                    R_loss = (R_criterion(R_outputs, y_source.float())
                            - hparams["lambda_"] * Dt_criterion(torch.squeeze(Dt_outputs), D_labels[disc_input_target]) * 2 * (Dt_outputs.shape[0]) / R_outputs.shape[0] 
                            - hparams["lambda_"] * Dv_criterion(torch.squeeze(Dv_outputs), D_labels[disc_input_valida]) * 2 * (Dv_outputs.shape[0]) / R_outputs.shape[0]  )
                  R_optimizer.zero_grad()
                  R_loss.backward()
                  R_optimizer.step()
                  if e%100 == 0:
                    print ('Tdisc_accuracy = '+str(accuracy(Dt_outputs, D_labels[disc_input_target])) + ' Vdisc_accuracy = '+str(accuracy(Dv_outputs, D_labels[disc_input_valida])))
                else:
                  D.train()
                  detached_weight = R.embed.weight.detach()
                #   print (torch.isnan(detached_weight).sum()/64)
                  D_outputs = D(detached_weight)
                #   print (torch.isnan(D_outputs).sum()/1)
                #   print (torch.isnan(D_labels).sum()/1)

                  D_loss = hparams["lambda_"] * D_criterion(torch.squeeze(D_outputs), D_labels)
                  D_optimizer.zero_grad()
                  D_loss.backward(retain_graph=True)
                  D_optimizer.step()
                  # Role-model optimization
                  R.train()
                  R_outputs = R(X_source)
                  move_weight = R.embed.weight
                  D_outputs = D(move_weight)
                  R_loss = R_criterion(R_outputs, y_source.float()) - hparams["lambda_"] * D_criterion(torch.squeeze(D_outputs), D_labels)
                  R_optimizer.zero_grad()
                  R_loss.backward()
                  R_optimizer.step()
                  if e%100 == 0:
                    print (e)
                    print ('disc_accuracy = '+str(get_accuracy(D_outputs, D_labels)))
                #   print(e, D_loss.item(), R_loss.item())#,hamming_loss(R_outputs,y_source.float()), one_errors(R_outputs,y_source.float()))
              # validation epoch
              R.eval()
              if isinstance(R, RoleModel):
                source_outputs = R(X_source)
                target_outputs = R(X_target)
                source_threshold = torch.ones_like(source_outputs)/2
                target_threshold = torch.ones_like(target_outputs)/2
              elif isinstance(R, RoleModel_ibpmll):
                source_outputs,source_threshold = split_tensor(R(X_source))
                target_outputs,target_threshold = split_tensor(R(X_target))
              
              counter = counter + 1
              r_list += [counter, randomseed]
              r_list += list(hparams.values())
              r_list += list(get_multi_metrix(source_outputs.cpu(), y_source.cpu(), source_threshold.cpu()))
              if is_val_label:
                val_outputs = target_outputs[:val_size]
                target_outputs = target_outputs[val_size:]
                val_threshold = target_threshold[:val_size]
                target_threshold = target_threshold[val_size:]
                r_list += list(get_multi_metrix(val_outputs.cpu(), y_val.cpu(), val_threshold.cpu()))
              else:
                r_list += [0]*7 # TODO
              if is_target_label:
                r_list += list(get_multi_metrix(target_outputs.cpu(), y_target.cpu(), target_threshold.cpu()))
              else:
                r_list += [0]*7 # TODO

              if save_para:
                present_V_F = ml_F(val_outputs.cpu(), y_val.cpu(), val_threshold.cpu())
                if TRAIN_DUMP_MAX_ONLY:
                    if max_val_accuracy >= present_V_F:
                        continue
                    else:
                        max_val_accuracy = present_V_F                
                after_target_X = R.embed.weight.detach()[:target_size, :].cpu().numpy()
                after_source_X = R.embed.weight.detach()[target_size:, :].cpu().numpy()
                target_outputs = target_outputs.detach().cpu().numpy()
                source_outputs = source_outputs.detach().cpu().numpy()
                target_threshold = target_threshold.detach().cpu().numpy()
                source_threshold = source_threshold.detach().cpu().numpy()
                if  is_val_label:
                    val_outputs = val_outputs.detach().cpu().numpy()
                    val_threshold = val_threshold.detach().cpu().numpy()
                else:
                    val_outputs = target_outputs
                    val_threshold = target_threshold
                save_params('para'+str(randomseed), str(counter), after_source_X, after_target_X, val_outputs, target_outputs, source_outputs, val_threshold, target_threshold, source_threshold)
  r_list_tmp = np.array(r_list).reshape(counter,-1)
  np.savetxt(HOME+'ALL_'+str(randomseed)+'_result.txt', r_list_tmp, fmt='%.5f')

def train_gs_modified_r2_LB(is_target_label, randomseed, id, is_visualize, is_validation, save_para=False):# para1:is_target_label 2:randomseed 3:id 4:is_visualize 5:is_val_label
  main(is_target_label, randomseed, id, is_visualize, is_validation, save_para)

"""#### Main"""

# for i in 005 010 015 020 025 030 035 040 045 050 055 060 065 070 075 080 085 090 095 100
DIR=os.getcwd()+"/workspace/"
print (DIR)

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=10000)

# PROJECK="wikipedia0"
TRAIN_SAVE = True
# trimNo = 1

for x in range(0,1):
  PROJECK = args[1]#+str(x)
  PRO=PROJECK+"/"
#   TRAIN_TRIMNO = 1 # for eliminate joint 
  TRAIN_TRIMNO = 0 # for eliminate joint 
#   TRAIN_TPYE = args[6] # sigle, multi, regression
#   TRAIN_TARGET = args[7] # N, L
  TRAIN_METHOD = 'pm'#args[8]
  TRAIN_SAVE = True
  TRAIN_DUMP_MAX_ONLY = True
  TRAIN_LOSS = args[6]#'ibpmll' # bce # bpmll
  TRAIN_DISCRIMINATOR = args[7] # 'd2': source & target; 'd3': source & target & validation
  TRAIN_IMB = args[8] # MLSMOTE # other
  TRAIN_MULTIDISCRIMINATOR = True if args[9] == 'multi_disc' else False
  TRAIN_DISCRIMINATOR_LOSS = args[10] # 'half':0.5*La + 0.5*Lb, 'nomalized': divided by nums
  TRAIN_OBJECT_EMB = args[11] # s2v, R2V, GW

  # for j in range(1,6,1):
  HOME=DIR+PROJECK+"_pm_v"+str(args[2])+"1_N_{}_3metrix_dic_{}_{}_{}_{}_NN_{}/".format(TRAIN_LOSS, TRAIN_DISCRIMINATOR, TRAIN_IMB, args[9], args[10],TRAIN_OBJECT_EMB)
#   HOME=DIR+PROJECK+"_pm_v"+str(args[2])+"1_N_bpmll_3metrix_r1/"
  if os.path.exists(HOME):
    import shutil
    shutil.rmtree(HOME)
  os.makedirs(HOME)
  i=1

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  translator(DIR+PRO+PROJECK+"_"+args[3]+".txt", DIR+PRO+PROJECK+"_ValTar_"+TRAIN_OBJECT_EMB+".emb", DIR+PRO+PROJECK+"_component_ValTar.txt", DIR+PRO+PROJECK+"_"+args[4]+".txt",  DIR+PRO+PROJECK+"_"+args[5]+".txt", TRAIN_DISCRIMINATOR ) # para1:source_label, p2:emb, p3:component, p4:target_label, p5:validation_label
#   translator(DIR+PRO+PROJECK+"_"+args[3]+".txt", DIR+PRO+PROJECK+"_ValTar.emb", DIR+PRO+PROJECK+"_component_ValTar.txt", DIR+PRO+PROJECK+"_"+args[4]+".txt",  None, TRAIN_DISCRIMINATOR ) # para1:source_label, p2:emb, p3:component, p4:target_label, p5:validation_label
  for j in range(int(args[2]),int(args[2])+10,1):
    train_gs_modified_r2_LB( True, j, "{:03}".format(i), True, True, save_para=TRAIN_SAVE) # para1:is_target_label 2:randomseed 3:id 4:is_visualize 5:is_validation
    # train_gs_modified_r2_LB( True, j, "{:03}".format(i), True, False, save_para=TRAIN_SAVE) # para1:is_target_label 2:randomseed 3:id 4:is_visualize 5:is_validation

