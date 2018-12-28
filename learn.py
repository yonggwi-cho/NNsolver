#!/usr/bin/env python

import numpy as np

import argparse
import sys

#for nnabla
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mat",type=str)
parser.add_argument("--rhs",type=str)
parser.add_argument("--sol",type=str)
parser.add_argument("--batch",type=int,default=10)
parser.add_argument("--num",type=int,default=1000)
parser.add_argument("--max_iter",type=int,default=1000)
parser.add_argument("--weight_decay",default=0.99)
parser.add_argument("--learning_rate",default=1.0e-3)
args = parser.parse_args()

#read size
def get_size(path_mat):
    mat = np.loadtxt(path_mat,comments="%")
    n = int(mat[0][0])
    nnz = int(mat[0][2])
    return n,nnz

# read a data
def read_a_data(path_mat,path_rhs):
    # read matrix
    mat = np.loadtxt(path_mat,comments="%")
    n = int(mat[0][0])
    nnz = int(mat[0][2])
    mat = mat[1:,2] # omit the first line

    # read rhs
    rhs = np.loadtxt(path_rhs,comments="%")
    n_check = int(rhs[0][0])
    rhs = rhs[1:,2] # omit the first line

    # check
    if n != n_check :
        print "matrix size and vector size are different! please confirm..."
        sys.exit(1)

    data = np.concatenate((mat,rhs))
    return data

#read data
def read_data(n_batch,n,data):
    icount = 0
    for i in iter(np.random.choice(n,n_batch)):
        temp = read_a_data(args.mat+str(i)+".mtx",args.rhs+str(i)+".mtx")
        for j in range(temp.size):
            data[icount][j] = temp[j]
        icount += 1

# read a solution vector
def read_a_solution(path_sol):
    data = np.loadtxt(path_sol,comments="%")
    data = data[1:,2] # omit the first line
    return data

# read solution vectors
def read_solution(n_batch,n,data):
    icount = 0
    for i in iter(np.random.choice(n,n_batch)):
        temp = read_a_solution(args.sol+str(i)+".mtx")
        for j in range(temp.size):
             data[icount][j] = temp[j]
        icount += 1
    #sol = list()
    #for i in range(args.num):
    #    sol = np.concatenate((sol,read_a_solution(args.sol+str(i)+".mtx")))
    #return sol


# define neural network
def network(x,n,nnz,test=False):
    # input layer
    with nn.parameter_scope('Affine'):
        h = PF.affine(x,n+nnz)
    with nn.parameter_scope("BatchNormalization"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    # hidden layer 1
    with nn.parameter_scope('Affine1'):
        h = PF.affine(h,n+nnz)
    with nn.parameter_scope("BatchNormalization1"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    # hidden layer 2
    with nn.parameter_scope('Affine2'):
        h = PF.affine(h,(n+nnz)/2)
    with nn.parameter_scope("BatchNormalization2"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    # hidden layer 3
    with nn.parameter_scope('Affine3'):
        h = PF.affine(h,n+nnz)
    with nn.parameter_scope("BatchNormalization3"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    # output layer
    with nn.parameter_scope('Affine4'):
        h = PF.affine(h,n)
    with nn.parameter_scope("BatchNormalization4"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    return h

#training
def train():
    n,nnz = get_size(args.mat+"0.mtx")
    print "(n,nnz,n+nnz)=",n,nnz,n+nnz
    x   = nn.Variable([args.batch, n+nnz])
    t   = nn.Variable([args.batch, n])
    y = network(x,n,nnz)
    loss = F.mean(F.huber_loss(y,t))

    # define np array
    x_data = np.zeros((args.batch,n+nnz))
    t_data = np.zeros((args.batch,n))

    # read data
    read_data(args.batch,args.num,x_data)
    read_solution(args.batch,args.num,t_data)
    x.d = x_data
    t.d = t_data
    loss.forward()
    print "Init loss = ", loss.d

    # print
    print nn.get_parameters()

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # optimise
    for i in range(args.max_iter):
        read_data(args.batch,args.num,x_data)
        read_solution(args.batch,args.num,t_data)
        x.d = x_data
        t.d = t_data
        loss.forward()
        solver.zero_grad()  # Initialize gradients of all parameters to zero.
        loss.backward()
        solver.weight_decay(1e-5)  # Applying weight decay as an regularization
        solver.update()
        #if i % 100 == 0:  # Print for each 10 iterations
        print i, loss.d

if __name__ == "__main__" :
    # train
    train()



