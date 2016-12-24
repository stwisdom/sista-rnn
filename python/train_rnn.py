#! /usr/bin/env python

"""
dosomething: a command line program that does something

Put a more elaborate description and other notes here.

Then do yourself a favor and include these two lines:

The general structure of this program is based on a template that can be found at
http://www.jamesstroud.com/jamess-miscellaneous-how-tos/python/cli-program-skeleton
"""

######################################################################
# import the next four no matter what
######################################################################
import os
import sys
import textwrap
from optparse import OptionParser

######################################################################
# for config file parsing, if needed
######################################################################
import yaml

######################################################################
# import other libraries and modules that do stuff
######################################################################
import pdb
from fftconv import cufft,cuifft
import cPickle
import numpy as np
import theano
import theano.tensor as T
from models import *
from optimizations import *
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,TimeDistributed
from keras.optimizers import RMSprop
import timeit, time
import os
import scipy
import scipy.linalg
import scipy.io as sio
import tables
import caltech
import cv2
import copy

######################################################################
# edit next two to your liking
######################################################################
__program__ = "run_rnn"
__version__ = "0.1"

######################################################################
# the defaults for the config, if needed
######################################################################
DEFAULTS = {"model" : 'uRNN_full',
            "loss_function" : "MSE",
            "alph" : 1.0,
            "lam1" : 0.5147,
            "lam2" : 1.0,
            "Vnorm" : 0.0,
            "Unorm" : 0.0,
            "flag_broadcast_silo" : True,
            "flag_connect_input_to_layers" : True,
            "K" : 3,
            "seed" : 2016,
            "learning_rate" : 0.0001,
            "batch_size" : 50,
            "downsample_train" : 1,
            "num_allowed_test_inc" : 200,
            "niter" : 10000,
            "savefile" : 'results_K3_lr0-0001',
            "flag_random_init" : 0,
            "flag_oracle_s1" : 0,
            "s1file" : None,
            "flag_train_sista_params" : 0
           }

######################################################################
# no need to touch banner() or usage()
######################################################################
def banner(width=70):
  hline = "=" * width
  sys.stderr.write(hline + "\n")
  p = ("%s v.%s " % (__program__, __version__)).center(width) + "\n"
  sys.stderr.write(p)
  sys.stderr.write(hline + "\n")

def usage(parser, msg=None, width=70, pad=4):
  lead_space = " " * (pad)
  w = width - pad
  err = ' ERROR '.center(w, '#').center(width)
  errbar = '#' * w
  errbar = errbar.center(width)
  hline = '=' * width
  if msg is not None:
    msg_list = str(msg).splitlines()
    msg = []
    for aline in msg_list:
      aline = lead_space + aline.rstrip()
      msg.append(aline)
    msg = "\n".join(msg)
    print '\n'.join(('', err, msg, errbar, ''))
    print hline
  print
  print parser.format_help()
  sys.exit(0)

######################################################################
# set up the options parser from the optparse module
#   - see http://docs.python.org/library/optparse.html
######################################################################
def doopts():

  ####################################################################
  # no need to edit the next line
  ####################################################################
  program = os.path.basename(sys.argv[0])

  ####################################################################
  # edit usg to reflect the options, usage, and info for user
  #   - see http://en.wikipedia.org/wiki/Backus-Naur_Form 
  #       - expressions in brackets are optional
  #       - expressions separated by bars are alternates
  #   - don't mess with the "%s", this is a template string
  #   - here, CSVFILE and associated usage info is just an example
  ####################################################################
  usg = """\
        usage: %s -h | -t | [-c CONFIG] MATFILE

          - MATFILE holds the data in MATLAB v7.3 .mat format

          - Use the -h flag to print this help
          - Use the -c flag to specify a config file in yaml format
          - Use the -t flag to output a template config to sdtout
        """
  usg = textwrap.dedent(usg) % program
  parser = OptionParser(usage=usg)

  ####################################################################
  # - these are only some examples
  # - but -t and -c options are recommended if using a config file
  ####################################################################  
  parser.add_option("-t", "--template", dest="template",
                    default=False, action="store_true",
                    help="print template settings file",
                    metavar="TEMPLATE")
  parser.add_option("-c", "--config", dest="config",
                    metavar="CONFIG", default=None,
                    help="config file to further specify conversion")
  return parser

######################################################################
# creates an easy configuration template for the user
######################################################################
def template():
  ####################################################################
  # this is yaml: http://www.yaml.org/spec/1.2/spec.html
  #   - keep the first document type line ("%YAML 1.2") and
  #     the document seperator ("---")
  #   - edit the template to match DEFAULTS
  ####################################################################
  t = """
      %YAML 1.2
      ---
      model : 'uRNN_full'
      loss_function : 'MSE'
      alph : 1.0
      lam1 : 0.5147
      lam2 : 1.0
      Vnorm : 0.0
      Unorm : 0.0
      flag_broadcast_silo : True
      K : 3
      seed : 2016
      learning_rate : 0.0001
      batch_size : 50
      downsample_train : 1
      num_allowed_test_inc : 200
      niter : 10000
      savefile : 'results_K3_lr0-0001'
      flag_random_init : 0
      flag_oracle_s1 : 0
      s1file : None
      """

  ####################################################################
  # no need to edit the next two lines
  ####################################################################
  print textwrap.dedent(t)
  sys.exit(0)

######################################################################
# helper functions
######################################################################
def write_img_files(x,imgfiles):
  # assume x is nrows x ncols x nfiles
  for i,imgfile in enumerate(imgfiles):
      d = os.path.dirname(imgfile)
      if not os.path.exists(d):
          os.makedirs(d)
      cv2.imwrite(imgfile,x[:,:,i])

class LossHistory(keras.callbacks.Callback):
    def __init__(self, histfile):
        self.histfile=histfile
    
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_acc  = []
        self.val_loss   = []
        self.val_acc    = []

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        cPickle.dump({'train_loss' : self.train_loss, 'train_acc' : self.train_acc, 'val_loss': self.val_loss, 'val_acc' : self.val_acc}, open(self.histfile, 'wb'))     

######################################################################
# process the command line logic, parse the config, etc.
######################################################################
def main():

  ####################################################################
  # no need to touch the next two lines
  ####################################################################
  parser = doopts()
  (options, args) = parser.parse_args()

  ####################################################################
  # start processing the command line logic here
  #   - this logic is for
  #     "usage: %s -h | -t | [-c CONFIG] CSVFILE"
  #   - don't touch the next four if you have a config file
  ####################################################################
  if options.template:
    template()
  else:
    banner()

  ####################################################################
  # just as an example, we get the value of 'csvfile' from args
  ####################################################################
  if len(args) != 1:
    usage(parser)
  else:
    mat_data = args[0]

  ####################################################################
  # create the configuration that will be used within the program
  #   - first make a copy of DEFAULTS
  #   - then, update the copy with the user_config of
  #     the config file
  #   - this allows the user to specify only a subset of the config
  #   - try to catch problems with the config file and
  #     report them in usage()
  ####################################################################
  config = DEFAULTS.copy()
  if options.config:
    if os.path.exists(options.config):
      f = open(options.config)
      user_config = yaml.load(f.read())
      config.update(user_config)
    else:
      msg = "Config file '%s' does not exist." % options.config
      usage(parser, msg)

  ####################################################################
  # Put the stuff your program does within a try-except block.
  # This block catches a top-level module exception that
  # sentinels errors that you anticipate and want the program to
  # handle gracefully.
  #
  # Here, dostuff.something() process the mythical 'csvfile'.
  # It takes the 'csvfile and the 'config' dictionary as arguments.
  ####################################################################
  #try:

  if ('dataset' in config) and (config['dataset']=='caltech256'):
      print "Using caltech256 dataset"
      print "Using matrices from file %s" % config['path_matrices']
      matrices = tables.openFile(config['path_matrices'])
      ydiv=np.float32(3.0)
      A=np.asarray(matrices.root.A[:]).astype(np.float32).transpose()
      D=np.asarray(matrices.root.D[:]).astype(np.float32).transpose()
      Dinv=np.asarray(matrices.root.Dinv[:]).astype(np.float32).transpose()
      F=np.eye(config['N']).astype(np.float32)
      # make sure the matrix norm of AD is less than 1, so that ISTA
      # will converge with fixed step size \alpha=1 [Daubschies et al. 2004]
      A=A/ydiv
      if not os.path.isfile('caltech256_N%s_color%s'%(config['N'],config['color'])):
          # load the images as y, which will be nexamples x nrows x ncols
          y, yfiles,_,_ = caltech.load_data((config['N'],config['N']),config['path_dataset'],color=config['color'])#[0:2]
          cPickle.dump({'y':y,'yfiles':yfiles},file('caltech256_N%s_color%s'%(config['N'],config['color']),'wb'),cPickle.HIGHEST_PROTOCOL) 
      
      data = cPickle.load(file('caltech256_N%s_color%s'%(config['N'],config['color']),'rb')) 
      
      y=data['y']
      yfiles=data['yfiles']
      rng=np.random.RandomState(2013)
      # shuffle the images
      idx_yshuffle=rng.permutation(y.shape[0])
      y = y[idx_yshuffle,:,:]
      yfiles_shuffle = [yfiles[i] for i in idx_yshuffle]
      yfiles = yfiles_shuffle
      # make y nrows x ncols x nexamples
      y = np.transpose(y,[1,2,0])
      # try to make columns of y zero-mean
      ymean = np.mean(y,axis=0)
      mean_ymean = np.mean(ymean)
      y = y-mean_ymean
      # scale y to have the specified mean standard deviation,
      # which is set to match the standard deviation of the
      # piecewise synthetic signal
      ystd = np.std(y,axis=0)
      mean_ystd = np.mean(ystd)
      y = y/mean_ystd
      y = y*config['std_cols']
      # create compressed observations
      #
      # matrix multiply between A and y,
      # MxN times nrows x ncols x nexamples
      # yields M x nrows x nexamples
      #
      # need to transpose y to get the right shape:
      # matrix multiply between A and y.transpose([1,0,2])
      # MxN times ncols x nrows x nexamples
      # yields M x ncols x nexamples
      x = np.dot(A,y.transpose([1,0,2]))
      # make x of shape ncols x nexamples x M=N/R
      x = np.transpose(x,[1,2,0])
      # make y of shape ncols x nexamples x N=nrows
      y = np.transpose(y,[1,2,0])
      # build train, validation, and test sets
      itrain0=0
      itrain1=np.int(np.floor(0.8*y.shape[1]))
      ivalid0=itrain1
      ivalid1=np.int(np.floor(0.9*y.shape[1]))
      itest0 =ivalid1
      itest1 =y.shape[1]
      ytrain=y[:,itrain0:itrain1,:]
      yvalid=y[:,ivalid0:ivalid1,:]
      ytest =y[:,itest0 :itest1 ,:]
      print "ytrain shape is",ytrain.shape
      print "yvalid shape is",yvalid.shape
      print "ytest shape is", ytest.shape
      xtrain=x[:,itrain0:itrain1,:]
      xvalid=x[:,ivalid0:ivalid1,:]
      xtest =x[:,itest0 :itest1 ,:]
      print "xtrain shape is",xtrain.shape
      print "xvalid shape is",xvalid.shape
      print "xtest shape is", xtest.shape
      
      yfiles_test=yfiles[itest0:itest1]
      
      # downsample data if desired
      if 'downsample_train' in config:
          print "Downsampling training data by %d" % config['downsample_train']
          ytrain=ytrain[:,::config['downsample_train'],:]
          xtrain=xtrain[:,::config['downsample_train'],:]
  
  else:
      print "Using synthetic piecewise dataset from [Asif and Romberg 2014]"
      print "Loading data from file %s" % mat_data
      data = tables.openFile(mat_data)

      # convert data variables into numpy
      ydiv=np.float32(3.0)
      A=np.asarray(data.root.A[:]).astype(np.float32).transpose()
      A=A/ydiv
      D=np.asarray(data.root.D[:]).astype(np.float32).transpose()
      Dinv=D.transpose() #inverse is just the transpose, because D is an orthonormal wavelet basis
      F=np.asarray(data.root.F[:]).astype(np.float32).transpose()
      xtrain=np.asarray(data.root.xtrain[:]).astype(np.float32).transpose()
      xtrain=xtrain/ydiv
      ytrain=np.asarray(data.root.ytrain[:]).astype(np.float32).transpose()
      xtest=np.asarray(data.root.xtest[:]).astype(np.float32).transpose()
      xtest=xtest/ydiv
      ytest=np.asarray(data.root.ytest[:]).astype(np.float32).transpose()
       
      # downsample training and test data
      downsample_train=config['downsample_train']
      xtrain=xtrain[:,::downsample_train,:]
      ytrain=ytrain[:,::downsample_train,:]
      
      # handle oracle signal initialization
      if config['flag_oracle_s1']:
        print "Using oracle s_1, loading from file %s" % config['s1file']
        # load up the oracle signal
        s1data = tables.openFile(config['s1file'])
        s1 = np.asarray(s1data.root.s1[:]).astype(np.float32) #s1 will be shape [1,N]
        h1 = np.dot(D.transpose(),s1.transpose()).transpose() #h1 will be shape [1,N]

        ## clip data to 2:T
        #xtrain=xtrain[1:,:,:]
        #ytrain=ytrain[1:,:,:]
        #xtest=xtest[1:,:,:]
        #ytest=ytest[1:,:,:]
      
      # create validation and evaluation sets
      xvalid=xtest[:,100:200,:]
      yvalid=ytest[:,100:200,:]
      xtest=xtest[:,0:100,:]
      ytest=ytest[:,0:100,:]

  # get data shapes
  [time_steps,ntrain,M]=xtrain.shape
  N=ytrain.shape[-1]
  ntest=xtest.shape[1]
  print "T=%d, M=%d, N=%d, ntrain=%d, ntest=%d" % (time_steps,M,N,ntrain,ntest)

  # save data to matfile if a destination is specified
  if ('matfile_data' in config):
      if not os.path.isfile(config['matfile_data']):
          print "Saving data to matfile %s" % config['matfile_data']
          sio.savemat(config['matfile_data'],{'D':D,'A':A,'xtrain':xtrain,'ytrain':ytrain,'xvalid':xvalid,'yvalid':yvalid,'xtest':xtest,'ytest':ytest,'mean_ymean':mean_ymean,'mean_ystd':mean_ystd,'std_cols':config['std_cols'],'yfiles_test':yfiles_test})


  print "Building theano computational graphs..."
  print "Configuration:"
  for key in config:
      print '  %s : ' %key,config[key]
  model=config['model']
  loss_function=config['loss_function']
  alph=np.float32(config['alph'])
  lam1=np.float32(config['lam1'])
  lam2=np.float32(config['lam2'])
  Vnorm=np.float32(config['Vnorm'])
  Unorm=np.float32(config['Unorm'])
  K=config['K']
  seed=config['seed']
  n_input=M
  n_hidden=N
  n_output=N
  learning_rate=np.float32(config['learning_rate'])
  nbatch=config['batch_size']
  if config['savefile'] is None:
      savefile='results_default'
  else:
      savefile=config['savefile']
  if (model=='LSTM'):
    model=Sequential()
    model.add(LSTM(n_hidden,return_sequences=True,input_shape=(time_steps,n_input)))
    for kk in range(K):
        model.add(LSTM(n_hidden,return_sequences=True))
    model.add(TimeDistributed(Dense(n_output)))
    rmsprop = RMSprop(lr=learning_rate,clipnorm=np.float32(1.0))
    model.compile(loss='mse',
                  optimizer=rmsprop)
    history=LossHistory(config['histfile'])
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=savefile, verbose=1, save_best_only=True)
    earlystopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], verbose=1, mode='auto') 
    
    #change shape of xtrain and ytrain from [T,nexamples,dim] to [nexamples,T,dim]
    xtrain=np.transpose(xtrain,[1,0,2])
    ytrain=np.transpose(ytrain,[1,0,2])
    xvalid=np.transpose(xvalid,[1,0,2])
    yvalid=np.transpose(yvalid,[1,0,2])
    xtest =np.transpose(xtest ,[1,0,2])
    ytest =np.transpose(ytest ,[1,0,2])

    if ('savefile_init' in config):
        model.load_weights(config['savefile_init'])

    if (config['nepoch']>0):
        model.fit(xtrain, ytrain, batch_size=nbatch, nb_epoch=config['nepoch'],
                  verbose=1,
                  validation_data=(xvalid,yvalid),
                  callbacks=[history,checkpointer,earlystopping])
    yest_test = model.predict_on_batch(xtest)
    yest_test = np.transpose(yest_test,[1,0,2])
  elif ('uRNN' in model):
    if ('full' in model):
        Wimpl='full'
        idx_project=[]
        for kk in range(K-1):
            idx_project=idx_project+[5+kk*4,6+kk*5]
        idx_project=idx_project+[5+(K-1)*4]
        #if not config['flag_random_init']:
        idx_project=[]
    inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type='real',out_every_t=True, loss_function=loss_function,output_type='real',flag_feed_forward=False,flag_return_lin_output=True,flag_use_mask=False,Wimpl=Wimpl,lam=lam1,Vnorm=Vnorm,Unorm=Unorm,flag_return_hidden_states=True,n_layers=K,seed=seed,flag_connect_input_to_layers=config['flag_connect_input_to_layers'],flag_broadcast_silo=config['flag_broadcast_silo'])

    # allocate theano shared variables
    s_xtrain = theano.shared(xtrain,borrow=True)
    s_ytrain = theano.shared(ytrain,borrow=True)
    s_xvalid = theano.shared(xvalid,borrow=True)
    s_yvalid = theano.shared(yvalid,borrow=True)
    s_xtest  = theano.shared(xtest,borrow=True)
    s_ytest  = theano.shared(ytest,borrow=True)

    if config['flag_random_init']:
      # we are randomly initializing parameters, use xavier init of
      # Gaussian noise with variance 2/(n_in + n_out)
      def xavier_init(A,rng):
            return np.sqrt(2.0/np.prod(A.shape))*rng.randn(*A.shape).astype(np.float32)
      xform=xavier_init
    else:
      # we are initializing with SISTA parameters, so just pass the data through
      xform=lambda x,junk: x

    # initialize network parameters
    if config['flag_train_sista_params']:
        # use SISTA parameterization for the RNN
        #use theano.gof.graph.inputs([costs[0]]) to get inputs for MSE cost
        #use theano.gof.graph.clone_with_new_inputs([inputs_new]) to create new
        #graph, where inputs_new is the same as inputs except with all variables
        #in the list 'parameters' replaced with reparameterized versions.
        #Trainable parameters should now be SISTA parameters, so build a new
        #'parameters' list that contains these shared variables.
        #Will need to recompute gradients, using
        #gradients = T.grad(costs_new,parameters_new)
    
        # allocate SISTA params
        def create_parameter(x,name):
            return theano.shared(x,name)
        parameters_sista=[]
        parameters_sista_init={'D':D,'Dinv':Dinv,'A':A,'F':F,'alph':alph,'lam1':lam1,'lam2':lam2}
        if 'untied_sista_parameters' in config:
            untied_sista_parameters=config['untied_sista_parameters']
        else:
            untied_sista_parameters=[]
        parameters_sista_dict={}
        for name in parameters_sista_init.keys():
            if name in untied_sista_parameters:
                # this parameter is untied across layers, so initialize a separate
                # parameter for each layer
                xs=[]
                for k in range(K):
                    x=create_parameter(parameters_sista_init[name],name+('_L%d'%k))
                    xs.append(x)
                    parameters_sista.append(x)
                parameters_sista_dict.update({name:xs})
            else:
                # this parameter is tied across layers, so initialize a single parameter
                # and copy it across the layers
                xs=[]
                x=create_parameter(parameters_sista_init[name],name)
                parameters_sista.append(x)
                for k in range(K):
                    xs.append(x)
                parameters_sista_dict.update({name:xs})

        inputs_graph=theano.gof.graph.inputs(costs)
        #inputs_new=copy.deepcopy(inputs_graph)
        inputs_replace={}

        # input transform V
        Vequiv=T.dot(parameters_sista_dict['D'][0].transpose(),parameters_sista_dict['A'][0].transpose())
        Vequiv_aug=T.concatenate([Vequiv.transpose(),np.zeros([M,N]).astype(np.float32)],axis=1)
        #inputs_new[inputs_graph.index(parameters[0])]=Vequiv
        inputs_replace.update({parameters[0] : Vequiv_aug})
        if config['flag_connect_input_to_layers']:
            for kk in range(K-1):
                Vequiv=T.dot(parameters_sista_dict['D'][kk+1].transpose(),parameters_sista_dict['A'][kk+1].transpose())
                Vequiv_aug=T.concatenate([Vequiv.transpose(),np.zeros([M,N]).astype(np.float32)],axis=1)
                #inputs_new[inputs_graph.index(parameters[10+kk*5])]=Vequiv
                inputs_replace.update({parameters[10+kk*5] : Vequiv_aug})
        
        #cross-iteration matrix S
        if K>1:
          Sequiv_aug=[]
          for k in range(K-1):
              #Sequiv=xform(np.eye(N).astype(np.float32)-np.dot(D.transpose(),np.dot(np.dot(A.transpose(),A)+lam2*np.eye(N).astype(np.float32),D))/alph,rng_init)
              #Sequiv_aug=np.concatenate( [np.concatenate([Sequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),Sequiv.transpose()],axis=1)], axis=0)
              Sequiv=np.eye(N).astype(np.float32)-T.dot(parameters_sista_dict['D'][k].transpose(),T.dot(T.dot(parameters_sista_dict['A'][k].transpose(),parameters_sista_dict['A'][k])+parameters_sista_dict['lam2'][k]*np.eye(N).astype(np.float32),parameters_sista_dict['D'][k]))/parameters_sista_dict['alph'][k]
              Sequiv_aug.append(T.concatenate( [T.concatenate([Sequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),T.concatenate([np.zeros([N,N],dtype=np.float32),Sequiv.transpose()],axis=1)], axis=0))
          #parameters[6].set_value(Sequiv_aug)
          #inputs_new[inputs_graph.index(parameters[6])]=Sequiv_aug[0]
          inputs_replace.update({parameters[6] : Sequiv_aug[0]})
          if config['flag_connect_input_to_layers']:
              for kk in range(K-2):
                #parameters[11+kk*5].set_value(Sequiv_aug)
                #inputs_new[inputs_graph.index(parameters[11+kk*5])]=Sequiv_aug[kk+1]
                inputs_replace.update({parameters[11+kk*5] : Sequiv_aug[kk+1]})
          else:
              for kk in range(K-2):
                #parameters[10+kk*4].set_value(Sequiv_aug)
                #inputs_new[inputs_graph.index(parameters[10+kk*5])]=Sequiv_aug[kk+1]
                inputs_replace.update({parameters[10+kk*5] : Sequiv_aug[kk+1]})
        
        #recurrence matrix W
        Wequiv_aug=[]
        for k in range(K):
            Dcur=parameters_sista_dict['D'][k]
            Dinvcur=parameters_sista_dict['Dinv'][k]
            Dt=Dcur.transpose()
            DDt=T.dot(Dcur,Dt)
            DtD=T.dot(Dt,Dcur)
            Acur=parameters_sista_dict['A'][k]
            AtA=T.dot(Acur.transpose(),Acur)
            Fcur=parameters_sista_dict['F'][k]
            FD=T.dot(Fcur,Dcur)
            alphcur=parameters_sista_dict['alph'][k]
            lam1cur=parameters_sista_dict['lam1'][k]
            lam2cur=parameters_sista_dict['lam2'][k]
            if k==0:
                Wequiv=(np.float32(1.0)+(lam2cur/alphcur))*T.dot(Dinvcur,FD) \
                         -(np.float32(1.0)/alphcur)*T.dot(Dt,T.dot(T.dot(AtA,Dcur),T.dot(Dinvcur,FD))) \
                         -(lam2cur/alphcur)*T.dot(DtD,T.dot(Dinvcur,FD))
            else:
                Wequiv=(lam2cur/alphcur)*T.dot(Dinvcur,FD) 
            Wequiv_aug.append(T.concatenate( [T.concatenate([Wequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),T.concatenate([np.zeros([N,N],dtype=np.float32),Wequiv.transpose()],axis=1)], axis=0))

        #parameters[5].set_value(W1equiv_aug)
        #inputs_new[inputs_graph.index(parameters[5])]=Wequiv_aug[0]
        inputs_replace.update({parameters[5] : Wequiv_aug[0]})
        if config['flag_connect_input_to_layers']:
            for kk in range(K-1):
              #parameters[9+kk*5].set_value(Wequiv_aug)
              #inputs_new[inputs_graph.index(parameters[9+kk*5])]=Wequiv_aug[kk+1]
              inputs_replace.update({parameters[9+kk*5] : Wequiv_aug[kk+1]})
        else:
            for kk in range(K-1):
              #parameters[9+kk*4].set_value(Wequiv_aug)
              #inputs_new[inputs_graph.index(parameters[9+kk*4])]=Wequiv_aug[kk+1]
              inputs_replace.update({parameters[9+kk*4] : Wequiv_aug[kk+1]})
        
        #output transform U
        #parameters[1].set_value(np.concatenate([xform(D.transpose(),rng_init),np.zeros([N,N],dtype=np.float32)],axis=0))
        #inputs_new[inputs_graph.index(parameters[1])]=T.concatenate([parameters_sista_dict['D'][-1].transpose(),np.zeros([N,N],dtype=np.float32)],axis=0)
        inputs_replace.update({parameters[1] : T.concatenate([parameters_sista_dict['D'][-1].transpose(),np.zeros([N,N],dtype=np.float32)],axis=0)})
        
        #soft-thresholds
        hidden_bias_shape=parameters[2].get_value().shape
        #parameters[2].set_value(-lam1/alph+hidden_bias) # add some mild noise to bias mean
        #parameters[2].set_value(xform(-(lam1/alph)*np.ones_like(hidden_bias).astype(np.float32),rng_init))
        hidden_bias=[]
        for k in range(K):
            lam1cur=parameters_sista_dict['lam1'][k]
            alphcur=parameters_sista_dict['alph'][k]
            hidden_bias.append(-(lam1cur/alphcur)*T.ones(hidden_bias_shape))
        #inputs_new[inputs_graph.index(parameters[2])]=hidden_bias[0]
        inputs_replace.update({parameters[2] : hidden_bias[0]})
        if config['flag_connect_input_to_layers']:
            for kk in range(K-1):
              #parameters[7+kk*5].set_value(xform(-(lam1/alph)*np.ones_like(hidden_bias).astype(np.float32),rng_init))
              #inputs_new[inputs_graph.index(parameters[7+kk*5])]=hidden_bias[kk+1]
              inputs_replace.update({parameters[7+kk*5] : hidden_bias[kk+1]})
        else:
            for kk in range(K-1):
              #parameters[7+kk*4].set_value(xform(-(lam1/alph)*np.ones_like(hidden_bias).astype(np.float32),rng_init))
              #inputs_new[inputs_graph.index(parameters[7+kk*4])]=hidden_bias[kk+1]
              inputs_replace.update({parameters[7+kk*4] : hidden_bias[kk+1]})

        #initial hidden state
        if config['flag_oracle_s1']:
          h1_aug=np.concatenate([xform(h1,rng_init),np.zeros([1,N],dtype=np.float32)],axis=1)
          parameters[4].set_value(h1_aug)
          parameters_sista.append(parameters[4])
          if config['flag_connect_input_to_layers']:
              for kk in range(K-1):
                parameters[8+kk*5].set_value(h1_aug)
                parameters_sista.append(parameters[8+kk*5])
          else:
              for kk in range(K-1):
                parameters[8+kk*4].set_value(h1_aug)
                parameters_sista.append(parameters[8+kk*4])
        else:
          parameters[4].set_value(np.zeros([1,2*N],dtype=np.float32))
          parameters_sista.append(parameters[4])
          if config['flag_connect_input_to_layers']:
              for kk in range(K-1):
                parameters[8+kk*5].set_value(np.zeros([1,2*N],dtype=np.float32))
                parameters_sista.append(parameters[8+kk*5])
          else:
              for kk in range(K-1):
                parameters[8+kk*4].set_value(np.zeros([1,2*N],dtype=np.float32))
                parameters_sista.append(parameters[8+kk*4])
        if not ('h_0' in untied_sista_parameters):
            for k in range(K-1):
                # replace each h_0 in upper K-1 layers with the h_0 from the first layer
                inputs_replace.update({parameters_sista[-(k+1)] : parameters_sista[-K]})
            parameters_sista=parameters_sista[:-(K-1)] #clip off h_0 from upper layers
        
        costs_new=[]
        for cost in costs:
            costs_new.append(theano.clone(cost,replace=inputs_replace))
        costs=costs_new
        parameters=parameters_sista
        print "Optimizing SISTA parameters"
    else:
        # use RNN parameterization for the RNN
        rng_init=np.random.RandomState(4299)
        #input transform V
        Vequiv=xform(np.dot(D.transpose(),A.transpose()),rng_init)
        Vequiv_aug=np.concatenate([Vequiv.transpose(),np.zeros([M,N]).astype(np.float32)],axis=1)
        parameters[0].set_value( Vequiv_aug )
        if config['flag_connect_input_to_layers']:
            for kk in range(K-1):
              parameters[10+kk*5].set_value(Vequiv_aug)
          
        #cross-iteration matrix S
        if K>1:
          Sequiv=xform(np.eye(N).astype(np.float32)-np.dot(D.transpose(),np.dot(np.dot(A.transpose(),A)+lam2*np.eye(N).astype(np.float32),D))/alph,rng_init)
          Sequiv_aug=np.concatenate( [np.concatenate([Sequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),Sequiv.transpose()],axis=1)], axis=0)
          parameters[6].set_value(Sequiv_aug)
          if config['flag_connect_input_to_layers']:
              for kk in range(K-2):
                parameters[11+kk*5].set_value(Sequiv_aug)
          else:
              for kk in range(K-2):
                parameters[10+kk*4].set_value(Sequiv_aug)

        #recurrence matrix W
        Dt=D.transpose()
        DDt=np.dot(D,Dt)
        DtD=np.dot(Dt,D)
        AtA=np.dot(A.transpose(),A)
        FD=np.dot(F,D)
        #W1equiv=(np.float32(1.0)+(lam2/alph))*np.dot(Dt,FD) \
        #         -(np.float32(1.0)/alph)*np.dot(Dt,np.dot(np.dot(AtA,DDt),FD)) \
        #         -(lam2/alph)*np.dot(Dt,np.dot(DDt,FD))
        W1equiv=xform((np.float32(1.0)+(lam2/alph))*np.dot(Dinv,FD) \
                 -(np.float32(1.0)/alph)*np.dot(Dt,np.dot(np.dot(AtA,D),np.dot(Dinv,FD))) \
                 -(lam2/alph)*np.dot(DtD,np.dot(Dinv,FD)),rng_init)
        W1equiv_aug=np.concatenate( [np.concatenate([W1equiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),W1equiv.transpose()],axis=1)], axis=0)
        #SFequiv=np.dot(Sequiv,np.dot(D.transpose(),np.dot(F,D)))
          #W1equiv_aug=np.concatenate( [np.concatenate([SFequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),SFequiv.transpose()],axis=1)], axis=0)
        Wequiv=xform((lam2/alph)*np.dot(Dinv,FD),rng_init)
        Wequiv_aug=np.concatenate( [np.concatenate([Wequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),Wequiv.transpose()],axis=1)], axis=0)
        
        #Wequiv_aug=np.zeros([2*N,2*N],dtype=np.float32)

        parameters[5].set_value(W1equiv_aug)
        if config['flag_connect_input_to_layers']:
            for kk in range(K-1):
              parameters[9+kk*5].set_value(Wequiv_aug)
        else:
            for kk in range(K-1):
              parameters[9+kk*4].set_value(Wequiv_aug)

        #output transform U
        parameters[1].set_value(np.concatenate([xform(D.transpose(),rng_init),np.zeros([N,N],dtype=np.float32)],axis=0))
        #soft-thresholds
        hidden_bias=parameters[2].get_value()
        #parameters[2].set_value(-lam1/alph+hidden_bias) # add some mild noise to bias mean
        parameters[2].set_value(xform(-(lam1/alph)*np.ones_like(hidden_bias).astype(np.float32),rng_init))
        if config['flag_connect_input_to_layers']:
            for kk in range(K-1):
              parameters[7+kk*5].set_value(xform(-(lam1/alph)*np.ones_like(hidden_bias).astype(np.float32),rng_init))
        else:
            for kk in range(K-1):
              parameters[7+kk*4].set_value(xform(-(lam1/alph)*np.ones_like(hidden_bias).astype(np.float32),rng_init))
                
        #initial hidden state
        if config['flag_oracle_s1']:
          h1_aug=np.concatenate([xform(h1,rng_init),np.zeros([1,N],dtype=np.float32)],axis=1)
          parameters[4].set_value(h1_aug)
          if config['flag_connect_input_to_layers']:
              for kk in range(K-1):
                parameters[8+kk*5].set_value(h1_aug)
          else:
              for kk in range(K-1):
                parameters[8+kk*4].set_value(h1_aug)
        else:
          parameters[4].set_value(np.zeros([1,2*N],dtype=np.float32))
          if config['flag_connect_input_to_layers']:
              for kk in range(K-1):
                parameters[8+kk*5].set_value(np.zeros([1,2*N],dtype=np.float32))
          else:
              for kk in range(K-1):
                parameters[8+kk*4].set_value(np.zeros([1,2*N],dtype=np.float32))
        """
        else:
          #input transform V
          Vequiv=np.dot(D.transpose(),A.transpose())/alph
          Vequiv_aug=np.concatenate([Vequiv.transpose(),np.zeros([M,N]).astype(np.float32)],axis=1)
          parameters[0].set_value( Vequiv_aug )
          for kk in range(K-1):
            parameters[10+kk*5].set_value(Vequiv_aug)
          #cross-iteration matrix S
          if K>1:
            Sequiv=np.eye(N).astype(np.float32)-np.dot(D.transpose(),np.dot(np.dot(A.transpose(),A)+lam2*np.eye(N).astype(np.float32),D))/alph
            Sequiv_aug=np.concatenate( [np.concatenate([Sequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),Sequiv.transpose()],axis=1)], axis=0)
            parameters[6].set_value(Sequiv_aug)
            for kk in range(K-2):
              parameters[11+kk*5].set_value(Sequiv_aug)
          #recurrence matrix W
          Dt=D.transpose()
          DDt=np.dot(D,Dt)
          DtD=np.dot(Dt,D)
          AtA=np.dot(A.transpose(),A)
          FD=np.dot(F,D)
          #W1equiv=(np.float32(1.0)+(lam2/alph))*np.dot(Dt,FD) \
          #         -(np.float32(1.0)/alph)*np.dot(Dt,np.dot(np.dot(AtA,DDt),FD)) \
          #         -(lam2/alph)*np.dot(Dt,np.dot(DDt,FD))
          W1equiv=(np.float32(1.0)+(lam2/alph))*np.dot(Dinv,FD) \
                   -(np.float32(1.0)/alph)*np.dot(Dt,np.dot(np.dot(AtA,D),np.dot(Dinv,FD))) \
                   -(lam2/alph)*np.dot(DtD,np.dot(Dinv,FD))
          W1equiv_aug=np.concatenate( [np.concatenate([W1equiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),W1equiv.transpose()],axis=1)], axis=0)
          #SFequiv=np.dot(Sequiv,np.dot(D.transpose(),np.dot(F,D)))
          #W1equiv_aug=np.concatenate( [np.concatenate([SFequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),SFequiv.transpose()],axis=1)], axis=0)
          Wequiv=(lam2/alph)*np.dot(Dinv,FD) 
          Wequiv_aug=np.concatenate( [np.concatenate([Wequiv.transpose(),np.zeros([N,N],dtype=np.float32)],axis=1),np.concatenate([np.zeros([N,N],dtype=np.float32),Wequiv.transpose()],axis=1)], axis=0)
        
          #Wequiv_aug=np.zeros([2*N,2*N],dtype=np.float32)

          parameters[5].set_value(W1equiv_aug)
          for kk in range(K-1):
            parameters[9+kk*5].set_value(Wequiv_aug)
          #output transform U
          parameters[1].set_value(np.concatenate([D.transpose(),np.zeros([N,N],dtype=np.float32)],axis=0))
          #soft-thresholds
          hidden_bias=parameters[2].get_value()
          #parameters[2].set_value(-lam1/alph+hidden_bias) # add some mild noise to bias mean
          parameters[2].set_value(-(lam1/alph)*np.ones_like(hidden_bias).astype(np.float32))
          for kk in range(K-1):
            parameters[7+kk*5].set_value(-(lam1/alph)*np.ones_like(hidden_bias).astype(np.float32))
          #initial hidden state
          if config['flag_oracle_s1']:
            h1_aug=np.concatenate([h1,np.zeros([1,N],dtype=np.float32)],axis=1)
            parameters[4].set_value(h1_aug)
            for kk in range(K-1):
              parameters[8+kk*5].set_value(h1_aug)
          else:
            parameters[4].set_value(np.zeros([1,2*N],dtype=np.float32))
            for kk in range(K-1):
              parameters[8+kk*5].set_value(np.zeros([1,2*N],dtype=np.float32))
        """

    print "Compiling theano computational graphs..."
    num_batches = int(np.ceil( ntrain / np.float(nbatch)))
    index = T.iscalar('i')
    gradients = T.grad(costs[0], parameters) 
    updates, rmsprop = rms_prop(learning_rate, parameters, gradients, idx_project)
    givens = {inputs[0] : s_xtrain[:, nbatch * index : nbatch * (index + 1), :],
              inputs[1] : s_ytrain[:, nbatch * index : nbatch * (index + 1), :]}
    train = theano.function([index], [costs[0],costs[1]], givens=givens, updates=updates)
    
    givens_train = {inputs[0] : s_xtrain[:,::100,:], inputs[1] : s_ytrain[:,::100,:]}
    train_no_step = theano.function([],[costs[0], costs[1], costs[2], costs[3]], givens=givens_train)
    
    givens_valid = {inputs[0] : s_xvalid,
                    inputs[1] : s_yvalid}
    valid = theano.function([], [costs[0], costs[1], costs[2], costs[3], costs[5]], givens=givens_valid)
    givens_test = {inputs[0] : s_xtest,
                   inputs[1] : s_ytest}
    test = theano.function([], [costs[0], costs[1], costs[2], costs[3], costs[5]], givens=givens_test)

    #grads = theano.function([index], [gradients], givens=givens) 

    # if we're doing the caltech256 dataset, let's write out the reference and initial images first
    if ('dataset' in config) and (config['dataset']=='caltech256') and config['flag_evaluate_initial']:
      #write out reference images
      # undo scaling and demeaning for the reference images
      ytest = ytest/config['std_cols']
      ytest = ytest*mean_ystd
      ytest = ytest+mean_ymean
      # get and modify filenames for images
      yfiles_test_ref = [f.replace(config['path_dataset'],config['path_ref']) for f in yfiles_test]
      yfiles_test_ref_png = [f.replace('.jpg','.png') for f in yfiles_test_ref]
      # write reference images
      print "Writing reference images to %s" % config['path_ref']
      write_img_files(255.0*np.transpose(ytest,[2,0,1]),yfiles_test_ref_png)

      # evaluate the network with initial parameters
      loss_test_init, mse_test_init, yest_test_init, ht_hest_init, losst_test_init = test()
      #write out images from initial network parameters
      # undo scaling and demeaning for the estimated images:
      yest_test_init = yest_test_init/config['std_cols']
      yest_test_init = yest_test_init*mean_ystd
      yest_test_init = yest_test_init+mean_ymean
      # clip yest to be between 0.0 and 1.0
      yest_test_init[yest_test_init<0.0]=np.float32(0.0)
      yest_test_init[yest_test_init>1.0]=np.float32(1.0)
      # get filenames for images
      yfiles_test_recon_init = [f.replace(config['path_dataset'],config['path_recon_init']) for f in yfiles_test]
      yfiles_test_recon_init_png = [f.replace('.jpg','.png') for f in yfiles_test_recon_init]
      # write reconstructed images
      print "Writing images reconstructed with initial parameters to %s" % config['path_recon_init']
      write_img_files(255.0*np.transpose(yest_test_init,[2,0,1]),yfiles_test_recon_init_png)
      print "EVALUATION (INITIAL)"
      print "Loss :",loss_test_init
      print "MSE  :",mse_test_init
      print "PSNR :"
      print "SSIM :"
  
    # run training loop
    train_loss = []
    train_mse  = []
    train_time = []
    valid_loss = []
    valid_mse  = []
    valid_time = []
    test_loss = []
    test_mse  = []
    test_time = []
    parameter_norms = []
    best_params = [p.get_value() for p in parameters]
    best_rms = [r.get_value() for r in rmsprop]
    best_valid_loss = np.inf
    mse_old = np.inf
    num_valid_inc=0
    num_allowed_valid_inc = config['num_allowed_test_inc']
    niter=config['niter']
    iters_per_validCheck=num_batches
    shuffle_rng=np.random.RandomState(314)

    if 'initfile' in config:
        #initialize all training parameters using specified initfile
        print "Using file %s to initialize training" % config['initfile']
        L=cPickle.load(file(config['initfile'],'r'))
        for iparam in xrange(len(L['parameters'])):
            parameters[iparam].set_value(L['parameters'][iparam])
        for irms in xrange(len(L['rmsprop'])):
            rmsprop[irms].set_value(L['rmsprop'][irms])
        train_loss = L['train_loss']
        train_time = L['train_time']
        valid_loss = L['valid_loss']
        valid_time = L['valid_time']
        test_loss = L['test_loss']
        test_time = L['test_time']
        parameter_norms = L['parameter_norms']
        best_valid_yest = L['best_valid_yest']
        best_valid_ht = L['best_valid_ht']
        best_test_yest = L['best_test_yest']
        best_test_ht = L['best_test_ht']
        best_params = L['best_params']
        best_rms = L['best_rms']
        best_valid_loss = L['best_valid_loss']
        # run the rng enough times to get back to the original seed
        for i in xrange(len(train_loss)):
            inds=shuffle_rng.permutation(ntrain)

    for i in xrange(niter):
      if (i % num_batches == 0):
        # reshuffle batch indices
        inds = shuffle_rng.permutation(ntrain)
        s_xtrain.set_value(xtrain[:,inds,:])
        s_ytrain.set_value(ytrain[:,inds,:])

      if ( i % iters_per_validCheck==0):
        start_time=time.time()
        loss, mse, yest, ht, losst = valid()
        elapsed_time = time.time() - start_time
        print ""
        print "VALIDATION"
        print "Loss:",loss
        print "MSE :",mse
        print "Time:", elapsed_time
        if (mse>mse_old):
          print "Validation cost function increased over last validation check!"
          print ""
        mse_old=mse
        valid_loss.append(loss)
        valid_mse.append(mse)
        valid_time.append(elapsed_time)

        start_time=time.time()
        loss_test, mse_test, yest_test, ht_test, losst_test = test()
        elapsed_time_test = time.time() - start_time
        print "EVALUATION"
        print "Loss:",loss_test
        print "MSE :",mse_test
        print "Time:", elapsed_time_test
        print ""
        test_loss.append(loss_test)
        test_mse.append(mse_test)
        test_time.append(elapsed_time_test)
        
        if (i==0):
            best_valid_yest=yest
            best_valid_ht=ht
            best_test_yest=yest_test
            best_test_ht=ht_test

        if mse < best_valid_loss:
            best_params = [p.get_value() for p in parameters]
            best_rm = [r.get_value() for r in rmsprop]
            best_valid_loss = mse
            best_valid_yest = yest
            best_valid_ht = ht
            best_test_yest = yest_test
            best_test_ht = ht_test
        else:
            num_valid_inc=num_valid_inc+1
            print "No improvement in validation loss, %d of %d allowed " % (num_valid_inc,num_allowed_valid_inc)
            print ""
            if num_valid_inc==num_allowed_valid_inc:
                print "Number of allowed validation loss increments reached. Stopping training..."
                print ""
                break
       
        """
        start_time=time.time()
        loss_temp, mse_temp, yest_temp, ht_temp = train_no_step()
        elapsed_time_temp = time.time() - start_time
        print "TRAIN (INITIAL)"
        print "Loss:",loss_temp
        print "MSE :",mse_temp
        print "Time:", elapsed_time_temp
        print ""
        """

        norms_cur=[]
        for p in parameters:
          norms_cur.append(np.linalg.norm(p.get_value()))
        parameter_norms.append(norms_cur)
        
        #grads_cur = grads(i % num_batches)

        save_vals = {'parameters': [p.get_value() for p in parameters],
                     'parameter_norms': parameter_norms,
                     'rmsprop': [r.get_value() for r in rmsprop],
                     'train_loss': train_loss,
                     'train_time': train_time,
                     'valid_loss': valid_loss,
                     'valid_time': valid_time,
                     'best_valid_yest': best_valid_yest,
                     'best_valid_ht': best_valid_ht,
                     'test_loss': test_loss,
                     'test_time': test_time,
                     'best_test_yest': best_test_yest,
                     'best_test_ht': best_test_ht,
                     'best_params': best_params,
                     'best_rms': best_rms,
                     'best_valid_loss': best_valid_loss}
        
        cPickle.dump(save_vals,
                     file(savefile, 'wb'),
                     cPickle.HIGHEST_PROTOCOL)

      # take training step
      start_time=time.time()
      loss,mse=train(i % num_batches)
      elapsed_time = time.time() - start_time
      train_loss.append(loss)
      train_mse.append(mse)
      train_time.append(elapsed_time)
      print "Iteration:",i
      print "Loss:",loss
      print "MSE :",mse
      print "Time:",elapsed_time
      print ""

    # write out the reconstructed images using the best parameters, if using
    # caltech256 dataset and if desired
    if ('dataset' in config) and (config['dataset']=='caltech256') and config['flag_evaluate']:
      # load best parameters
      L=cPickle.load(file(savefile,'rb'))
      best_params=L['best_params']
      for i,p in enumerate(best_params):
          parameters[i].set_value(p)
         
      loss_test, mse_test, yest_test, ht_test, losst_test = test()
      print "EVALUATION (BEST)"
      print "Loss :",loss_test
      print "MSE  :",mse_test
      print "PSNR :"
      print "SSIM :"
  
  # undo scaling and demeaning for the estimated images:
  yest_test = yest_test/config['std_cols']
  yest_test = yest_test*mean_ystd
  yest_test = yest_test+mean_ymean
  # clip yest to be between 0.0 and 1.0
  yest_test[yest_test<0.0]=np.float32(0.0)
  yest_test[yest_test>1.0]=np.float32(1.0)
  # get filenames for images
  yfiles_test_recon_best = [f.replace(config['path_dataset'],config['path_recon_best']) for f in yfiles_test]
  yfiles_test_recon_best_png = [f.replace('.jpg','.png') for f in yfiles_test_recon_best]
  # write reconstructed images
  print "Writing images reconstructed with best validation parameters to %s" % config['path_recon_best']
  write_img_files(255.0*np.transpose(yest_test,[2,0,1]),yfiles_test_recon_best_png)

  #except dostuff.DoStuffError, e:
  #  usage(parser, e)

if __name__ == "__main__":
  main()

