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

# read all data
def read_data_all(n,data):
    print "reading all data..."
    for i in range(n):
        temp = read_a_data(args.mat+str(i)+".mtx",args.rhs+str(i)+".mtx")
        for j in range(temp.size):
            data[i][j] = temp[j]
    print "finished reading all data."

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

# read all solution
def read_solution_all(n,data):
    print "reading all solution..."
    for i in range(n):
        temp = read_a_solution(args.sol+str(i)+".mtx")
        for j in range(temp.size):
             data[i][j] = temp[j]
    print "finished reading all solution."

# get batch data
def get_batch(data,data_b,n,n_batch):
    icount = 0
    for ibatch in iter(np.random.choice(n,n_batch)):
        for j in range(data.shape[1]):
             data_b[icount][j] = data[ibatch][j]
        icount += 1

# define neural network
def network(x,n,nnz,test=False):
    # input layer
    with nn.parameter_scope('Affine'):
        h = PF.affine(x,n+nnz)
    with nn.parameter_scope("BatchNormalization"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    with nn.parameter_scope('Relu'):
        h = F.relu(h)
    # hidden layer 1
    with nn.parameter_scope('Affine1'):
        h = PF.affine(h,n+nnz)
    with nn.parameter_scope("BatchNormalization1"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    with nn.parameter_scope('Relu1'):
        h = F.relu(h)
    # hidden layer 2
    with nn.parameter_scope('Affine2'):
        h = PF.affine(h,(n+nnz)/2)
    with nn.parameter_scope("BatchNormalization2"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    with nn.parameter_scope('Relu2'):
        h = F.relu(h)
    # hidden layer 3
    with nn.parameter_scope('Affine3'):
        h = PF.affine(h,n+nnz)
    with nn.parameter_scope("BatchNormalization3"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    with nn.parameter_scope('Relu3'):
        h = F.relu(h)
    # output layer
    with nn.parameter_scope('Affine4'):
        h = PF.affine(h,n)
    with nn.parameter_scope("BatchNormalization4"):
        h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
    return h

#training
def train(args):
    n,nnz = get_size(args.mat+"0.mtx")
    print "(n,nnz)=",n,nnz

    # read all data
    x_all = np.zeros((args.num,n+nnz))
    t_all = np.zeros((args.num,n))
    read_data_all(args.num,x_all)
    read_solution_all(args.num,t_all)

    x   = nn.Variable([args.batch, n+nnz])
    t   = nn.Variable([args.batch, n])
    y   = network(x,n,nnz)
    loss = F.mean(F.huber_loss(y,t))

    # define np array
    x_data = np.zeros((args.batch,n+nnz))
    t_data = np.zeros((args.batch,n))

    # get batch data
    get_batch(x_all,x_data,args.num,args.batch)
    get_batch(t_all,t_data,args.num,args.batch)
    x.d = x_data
    t.d = t_data
    loss.forward()
    print "Init loss = ", loss.d

    # print
    print nn.get_parameters()

    for param in nn.get_parameters().values():
        param.grad.zero()
    loss.backward()

    # Create Solver.
    #solver = S.Adam(args.learning_rate)
    solver = S.Sgd(args.learning_rate)
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
        solver.weight_decay(args.weight_decay)  # Applying weight decay as an regularization
        solver.update()
        #if i % 100 == 0:  # Print for each 10 iterations
        print i, loss.d

if __name__ == "__main__" :
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat",type=str)
    parser.add_argument("--rhs",type=str)
    parser.add_argument("--sol",type=str)
    parser.add_argument("--batch",type=int,default=64)
    parser.add_argument("--num",type=int,default=1000)
    parser.add_argument("--max_iter",type=int,default=1000)
    parser.add_argument("--weight_decay",type=float,default=1.0e-5)
    parser.add_argument("--learning_rate",type=float,default=1.0e-3)
    parser.add_argument("-c","--c",type=str,default="cpu")
    parser.add_argument("--device_id",type=int,default=0)
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    args = parser.parse_args()
    
    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.c)
    ctx = get_extension_context(
        args.c, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    
    # train
    train(args)



