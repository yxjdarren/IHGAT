import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from torchsummary import summary
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.metrics import normalized_mutual_info_score,accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from numpy.random import shuffle

import util
from models.net import LHGNets

from dataset import MtvDataset
from metric import accuracy
from visualize import *
import warnings
warnings.filterwarnings("ignore")

def init_env(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) / y_pred.size

def f1_score(gt_s, s):
    N = len(gt_s)
    num_t = 0
    num_h = 0
    num_i = 0
    for n in range(N-1):
        tn = (gt_s[n] == gt_s[n+1:]).astype('int')
        hn = (s[n] == s[n+1:]).astype('int')
        num_t += np.sum(tn)
        num_h += np.sum(hn)
        num_i += np.sum(tn * hn)
    p = r = f = 1
    if num_h > 0:
        p = num_i / num_h
    if num_t > 0:
        r = num_i / num_t
    if p + r == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return f

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def parse_args():
    parser = argparse.ArgumentParser(
        description='latent heterogeneous graph network for incomplete multi-view learning')

    # experiment set
    parser.add_argument('--data', type=str, required=True, help='Dataset feed to model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU index for cuda used')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent representation dim')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat n times experiment')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--missing_rate', type=float, default=0.5, help='Missing rate for view-specfic data')
    parser.add_argument('--split_rate', type=float, default=0.8, help='Train and test data split ratio')
    parser.add_argument('--log_path', type=str, default='./log.txt', help='Log file save path')
    parser.add_argument('--normalize', type=bool, default=True, help='Normalization for dataset')
    parser.add_argument('--dropout_rate', type=float, default=0.6, help='Dropout rate(1-keep probability). ')
    parser.add_argument('--nheads', type=int, default=3, help='Head number for self-attention')
    parser.add_argument('--lamda', type=int, default=0.1, help='coefficient of clustering loss')
    parser.add_argument('--k', type=int, default=10, help='n_neighbors')


    args = parser.parse_args()
    args.logger = util.get_logger(args.log_path)
    # init environment: gpu, cuda
    init_env(args)
    args.logger.info(args)
    return args


def train(args, model, data, label, train_idx, feature_mask, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    # criterion = nn.CrossEntropyLoss()
    rec_vec, output, semantic,latent,p,q,meta_att_vec= model(feature_mask)
    label = label.long().view(-1, )

    # update target distribution p
    # cross_KL loss
    # q = q.data
    # p = target_distribution(q)
    cross_kl_loss = 0
    for v1 in range(args.view_num):
        for v2 in range(args.view_num):

            loss_tmp = F.kl_div(q[v1][train_idx].log(), p[v1][train_idx])
            # weight = weight_mask[:, v1].double()
            # loss = loss_tmp * weight
            cross_kl_loss += loss_tmp
            # args.logger.warning("View " + str(v) + " kl loss " + str(loss.item()))


    # vies-exsistence loss
    view_consistence_loss = 0.0
    for v1 in range(args.view_num):
        for v2 in range(args.view_num):
            sum = torch.sum(
                torch.pow(torch.sub(meta_att_vec[v1], meta_att_vec[v2]), 2.0), 1)
            fea = feature_mask[:, v1].double()
            loss = sum * fea
            loss = torch.sum(loss)
            # args.logger.warning("View " + str(v) + " loss " + str(loss.item()))
            view_consistence_loss += loss



    # classification loss
    cls_loss = F.nll_loss(output[train_idx], label[train_idx])
    # args.logger.warning("Classfication loss " + str(cls_loss.item()))

    #KL Loss
    # kl_loss = 1e7*F.kl_div(q.log(), p)
    args.logger.warning("cross_kl_loss " + str(cross_kl_loss.item()))

    # reconstruction loss
    rec_loss = 0.0
    for v in range(args.view_num):
        sum = torch.sum(
            torch.pow(torch.sub(rec_vec[v], data[v]), 2.0), 1)
        fea = feature_mask[:, v].double()
        loss = sum * fea
        loss = torch.sum(loss)
        # args.logger.warning("View " + str(v) + " loss " + str(loss.item()))
        rec_loss += loss

    args.gamma =100
    # loss = args.gamma * kl_loss + rec_loss
    loss = args.gamma *cross_kl_loss +rec_loss

    # loss = 100*cls_loss +rec_loss
    # loss = rec_loss

    # # summary loss
    # if epoch < 100:
    #     loss = rec_loss
    # else:
    #     loss = kl_loss + rec_loss
    args.logger.warning("Total loss " + str(loss.item()))

    loss.backward()
    optimizer.step()
    acc_train = 0
    # acc_train = accuracy(output[train_idx], label[train_idx]).item()
    # args.logger.error("Epoch : " + str(epoch) + ' train accuracy : ' + str(acc_train))
    return acc_train,meta_att_vec,loss

def test(args, model, data, label, test_idx, feature_mask, epoch=0):
    model.eval()
    with torch.no_grad():
        _, output,semantic,latent,p,q,meta_att_vec = model(feature_mask)
        # loss_test = F.nll_loss(output[test_idx], label[test_idx])
        # acc_test = accuracy(output[test_idx], label[test_idx]).item()

        # loss_test = F.nll_loss(output[test_idx], label[test_idx])
        test_data  = semantic[test_idx]
        test_data = test_data.cpu().detach().numpy()
        test_label = label[test_idx]
        test_label = test_label.cpu().detach().numpy()

        # evaluate clustering performance
        labels = label
        labels = labels.cpu().detach().numpy()
        n_clusters = len(np.unique(labels))
        # y_x = KMeans(n_clusters=n_clusters , random_state=0).fit_predict(test_data)
        y_x =q[0].cpu().numpy().argmax(1)
        y_x = y_x[test_idx]
        #ACC, NMI
        acc_x = cluster_acc(test_label, y_x)
        nmi = nmi_score(test_label, y_x)
        f_measure = f1_score(test_label, y_x)
        # ri = rand_index_score(test_label, y_x).round(4)
        ar = ari_score(test_label, y_x)
        print("accuracy: %.4f" % acc_x,\
              "nmi: %.4f" % nmi, \
              "F-measure: %.4f" % f_measure, \
              # "RI: %.4f" % ri, \
              "AR: %.4f" % ar
              )
        args.logger.error("Epoch : " + str(epoch) + ' test accuracy : ' + str(acc_x) + ' test nmi : ' + str(nmi) + ' test f_measure : ' + str(f_measure) + ' test ar : ' + str(ar))
        # args.logger.error("Epoch : " + str(epoch) + ' test accuracy : ' + str(acc_test))
    return acc_x,nmi


def main(args):
    # load data
    mtv_data = MtvDataset(args)
    train_data, train_label = mtv_data.get_data('train')
    test_data, test_label = mtv_data.get_data('test')
    view_num = mtv_data.view_number
    view_dim = [train_data[i].shape[1] for i in range(view_num)]
    args.view_num = view_num

    train_sample_number = train_data[0].shape[0]
    test_sample_number = test_data[0].shape[0]
    all_sample_number = train_sample_number + test_sample_number

    # get incomplete mask
    feature_mask = mtv_data.get_missing_mask()

    # transductive learning setting
    train_idx = np.arange(0, train_sample_number)
    test_idx = np.arange(train_sample_number, all_sample_number)
    shuffle(train_idx)
    label = torch.cat([train_label, test_label])
    label = label - 1
    label = label.long().view(-1, )
    cls_num = torch.max(label).item() + 1
    data = {}
    for v in range(view_num):
        data[v] = torch.cat([train_data[v], test_data[v]])

    # build model
    model = LHGNets(view_num, all_sample_number, view_dim, cls_num, args.dropout_rate, args.nheads, args.gpu,
                    latent_dim=args.latent_dim).double()

    # transfer cuda
    if args.gpu != '-1':
        label = label.cuda()
        model = model.cuda()
        feature_mask = feature_mask.cuda()
        for v in range(view_num):
            data[v] = data[v].cuda()

    args.logger.info("Model Parameters : ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            args.logger.info(str(name) + str(param.data.shape))

    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    acc_curve = []
    nmi_curve = []
    loss_curve = []

    for i in range(args.epochs):
        train_acc, meta_att_vec,loss= train(args, model, data, label, train_idx, feature_mask, optimizer, i)
        test_acc,test_nmi = test(args, model, data, label, test_idx, feature_mask, i)
        acc_curve.append(test_acc)
        nmi_curve.append(test_nmi)
        loss_curve.append(loss.data.cpu().numpy())
        # tsne_visualization
        # if i in [1, 20,30, 40, 50, 60,70,80,90,100,120,150,199]:
        #     fea_train = meta_att_vec[0]
        #     fea_train = fea_train.cpu().detach().numpy()
        #     la = label
        #     la = la.cpu().detach().numpy()
        #     visualize_data_tsne(fea_train, la, 10, './figures/' + args.data + '_tsne_clu' + str(i) + '.svg')
    print(acc_curve)
    print(nmi_curve)
    print(loss_curve)
    args.logger.error("acc_curve : " + str(acc_curve) + ' nmi_curve : ' + str(nmi_curve) + ' loss_curve : ' + str(loss_curve) )
    # plt.plot(acc, label='acc')
    # plt.plot(nmi, label='nmi')
    # plt.xlabel('epoch')
    # plt.ylabel('score')
    # plt.legend(loc='best')
    # plt.savefig(".\curve.png")


    return test_acc,test_nmi

def run():
    # init env
    args = parse_args()

    # repeat n times experiment for average
    acc = []
    nmi = []
    for i in range(args.repeat):
        test_acc,test_nmi = main(args)
        acc.append(test_acc)
        nmi.append(test_nmi)

    args.logger.info(args)

if __name__ == '__main__':
    run()
