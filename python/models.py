import theano, cPickle
import theano.tensor as T
import numpy as np
from fftconv import cufft, cuifft

def initialize_matrix(n_in, n_out, name, rng, init='rand'):
    if (init=='rand') or (init=='randSmall'):
        bin = np.sqrt(6. / (n_in + n_out))
        values = np.asarray(rng.uniform(low=-bin,
                                        high=bin,
                                        size=(n_in, n_out)),
                                        dtype=theano.config.floatX)
        if (init=='randSmall'):
            values=np.float32(0.01)*values
    elif (init=='identity'):
        if (n_in >= n_out):
            values = np.concatenate([np.eye(n_out).astype(theano.config.floatX),np.zeros((n_in-n_out,n_out)).astype(theano.config.floatX)],axis=0)
        else:
            values = np.concatenate([np.eye(n_in).astype(theano.config.floatX),np.zeros((n_in,n_out-n_in)).astype(theano.config.floatX)],axis=1)
    else:
       raise ValueError("Unknown initialization method ["+init+"]") 
    return theano.shared(value=values, name=name)

def initialize_matrix_np(n_in, n_out, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX)
    return values

def do_fft(input, n_hidden):
    fft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    fft_input = fft_input.dimshuffle(0,2,1)
    fft_output = cufft(fft_input) / T.sqrt(n_hidden)
    fft_output = fft_output.dimshuffle(0,2,1)
    output = T.reshape(fft_output, (input.shape[0], 2*n_hidden))
    return output

def do_ifft(input, n_hidden):
    ifft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    ifft_input = ifft_input.dimshuffle(0,2,1)
    ifft_output = cuifft(ifft_input) / T.sqrt(n_hidden)
    ifft_output = ifft_output.dimshuffle(0,2,1)
    output = T.reshape(ifft_output, (input.shape[0], 2*n_hidden))
    return output


def times_diag(input, n_hidden, diag, swap_re_im):
    # input is a Ix2n_hidden matrix, where I is number
    # of training examples
    # diag is a n_hidden-dimensional real vector, which creates
    # the 2n_hidden x 2n_hidden complex diagonal matrix using 
    # e.^{j.*diag}=cos(diag)+j.*sin(diag)
    d = T.concatenate([diag, -diag]) #d is 2n_hidden
    
    Re = T.cos(d).dimshuffle('x',0)
    Im = T.sin(d).dimshuffle('x',0)

    input_times_Re = input * Re
    input_times_Im = input * Im

    output = input_times_Re + input_times_Im[:, swap_re_im]
   
    return output
    
    
def vec_permutation(input, index_permute):
    return input[:, index_permute]      

    
def times_reflection(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]
   
    vstarv = (reflection**2).sum()
    
    input_re_reflect_re = T.dot(input_re, reflect_re)
    input_re_reflect_im = T.dot(input_re, reflect_im)
    input_im_reflect_re = T.dot(input_im, reflect_re)
    input_im_reflect_im = T.dot(input_im, reflect_im)

    a = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
    b = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
    c = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
    d = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)
         
    output = input
    output = T.inc_subtensor(output[:, :n_hidden], - 2. / vstarv * (a + b))
    output = T.inc_subtensor(output[:, n_hidden:], - 2. / vstarv * (d - c))

    return output    

def times_reflection_sub(input, n_hidden, n_sub, reflection):
    
    #print "n_hidden=%d, n_sub=%d" % (n_hidden,n_sub)    
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    n_start=n_hidden-n_sub
    #print "n_start=%d" % n_start
    reflect_re = reflection[n_start:n_hidden]
    reflect_im = reflection[(n_hidden+n_start):]
   
    vstarv = (reflect_re**2).sum() + (reflect_im**2).sum()
    
    input_re_reflect_re = T.dot(input_re[:,n_start:], reflect_re)
    input_re_reflect_im = T.dot(input_re[:,n_start:], reflect_im)
    input_im_reflect_re = T.dot(input_im[:,n_start:], reflect_re)
    input_im_reflect_im = T.dot(input_im[:,n_start:], reflect_im)

    a = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
    b = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
    c = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
    d = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)

    output = input
    output = T.inc_subtensor(output[:, n_start:n_hidden], - 2. / vstarv * (a + b))
    output = T.inc_subtensor(output[:, (n_hidden+n_start):], - 2. / vstarv * (d - c))

    return output    

def times_givens(input,n_hidden,gparams,idx):
    # input is a Ix2n complex matrix in augmented ReIm form.
    # gparams is a 3-dim vector parameterizing a Givens rotation.
    # idx are two indices of the Givens rotation.
    #
    # output will be Ix2n complex matrix in augmented ReIm form     
    
    # Givens rotation using gparams=(phi,psi,chi) is
    # [ cos(phi)*exp( j*psi),  sin(phi)*exp( j*chi) ]
    # [-sin(phi)*exp(-j*psi),  cos(phi)*exp(-j*chi) ]
    cos_phi=T.cos(gparams[0])
    sin_phi=T.sin(gparams[0])
    cos_psi=T.cos(gparams[1])
    sin_psi=T.sin(gparams[1])
    cos_chi=T.cos(gparams[2])
    sin_chi=T.sin(gparams[2])
  
    G11_re=cos_phi*cos_psi
    G11_im=cos_phi*sin_psi
    G12_re=sin_phi*cos_chi
    G12_im=sin_phi*sin_chi
    G21_re=-sin_phi*cos_chi
    G21_im= sin_phi*sin_chi
    G22_re= cos_phi*cos_psi
    G22_im=-cos_phi*sin_psi

    idx=T.cast(idx,'int64')

    output=input
    # Re{y_{i1}}=Re{ G11*x_{i1}+G12*x_{i2} }
    #output[:,idx[0]] 
    output = T.set_subtensor(output[:,idx[0]], \
	               (G11_re*input[:,idx[0]]-G11_im*input[:,idx[0]+n_hidden]) \
                       +(G12_re*input[:,idx[1]]-G12_im*input[:,idx[1]+n_hidden]))
    # Im{y_{i1}}=Im{ G11*x_{i1}+G12*x_{i2} }
    #output[:,idx[0]+n_hidden] 
    output = T.set_subtensor(output[:,idx[0]+n_hidden], \
	                        (G11_im*input[:,idx[0]]+G11_re*input[:,idx[0]+n_hidden]) \
                                +(G12_im*input[:,idx[1]]+G12_re*input[:,idx[1]+n_hidden]))
    # Re{y_{i2}}=Re{ G21*x_{i1}+G22*x_{i2} }
    #output[:,idx[1]] 
    output = T.set_subtensor(output[:,idx[1]], \
	               (G21_re*input[:,idx[0]]-G21_im*input[:,idx[0]+n_hidden]) \
                       +(G22_re*input[:,idx[1]]-G22_im*input[:,idx[1]+n_hidden]))
    # Im{y_{i2}}=Im{ G21*x_{i1}+G22*x_{i2} }
    #output[:,idx[1]+n_hidden] 
    output = T.set_subtensor(output[:,idx[1]+n_hidden], \
	                        (G21_im*input[:,idx[0]]+G21_re*input[:,idx[0]+n_hidden]) \
                                +(G22_im*input[:,idx[1]]+G22_re*input[:,idx[1]+n_hidden]))
    
    return output 

def compute_cost_t(lin_output, loss_function, y_t, ymask_t=None, z_t=None, lam=0.0):
    if (loss_function == 'CE') or (loss_function == 'CE_of_sum'):
        RNN_output = T.nnet.softmax(lin_output)
        CE = T.nnet.categorical_crossentropy(RNN_output, y_t)
        if ymask_t is not None:
            RNN_output=RNN_output*ymask_t
            CE = CE*(ymask_t.dimshuffle(0,))
        cost_t = CE.mean()
        acc_t =(T.eq(T.argmax(RNN_output, axis=-1), y_t)).mean(dtype=theano.config.floatX)
    elif loss_function == 'MSE':
        mse = (lin_output - y_t)**2
        if ymask_t is not None:
            mse = mse*((ymask_t[:,0]).dimshuffle(0,'x'))
            #mse = mse*ymask_t[:,0:1]
        cost_t = mse.mean()
        #acc_t = theano.shared(np.float32(0.0))
        acc_t = cost_t
    elif loss_function == 'MSEplusL1':
        mseOnly = (lin_output - y_t)**2
        L1 = T.sqrt(1e-5 + T.sum(lin_output**2,axis=1,keepdims=True))
        mse = mseOnly + lam*L1
        if ymask_t is not None:
            mse = mse*((ymask_t[:,0]).dimshuffle(0,'x'))
        cost_t = mse.mean()
        acc_t = mseOnly.mean()
    #elif loss_function == 'NMSE':
    #    err=(lin_output - y_t)**2
    #    err_sum=T.sum(err,axis=0)
    #    err_sum=T.sum(err_sum,axis=-1)
    #    ypow=y_t**2
    #    ypow_sum=T.sum(ypow,axis=0)
    #    ypow_sum=T.sum(ypow_sum,axis=-1)
    #    cost_t = (err_sum / (1e-5+ypow_sum)).mean()
    #    acc_t = theano.shared(np.float32(0.0))
    elif (loss_function == 'g_loss') or (loss_function == 'none_in_scan'):
        cost_t=theano.shared(np.float32(0.0))
        acc_t =theano.shared(np.float32(0.0))
    elif loss_function == 'd_loss':
        RNN_output = T.nnet.sigmoid(lin_output)
        # clip the output of the sigmoid to avoid 0s, and thus NaNs in cross entropy:
        RNN_output_clip = T.clip(RNN_output,1e-7,1.0-1e-7)
        costs_t = T.nnet.binary_crossentropy(RNN_output_clip, y_t)
        if ymask_t is not None:
            costs_t = costs_t*(ymask_t.dimshuffle(0,))
        cost_t = costs_t.mean()
        idx_half=costs_t.shape[0]/2
        costs_t_fake=costs_t[:idx_half]
        costs_t_real=costs_t[idx_half:]
        acc_t = [costs_t_fake.mean()/2,costs_t_real.mean()/2]

    return cost_t, acc_t


def initialize_data_nodes(loss_function, input_type, out_every_t):
    # if input_type is real or complex, will be size n_fram x n_input x n_utt
    x = T.tensor3() if input_type == 'real' or input_type == 'complex' else T.matrix(dtype='int32')
    if 'CE' in loss_function:
        y = T.matrix(dtype='int32') if out_every_t else T.vector(dtype='int32')
    else:
        # y will be n_fram x n_output x n_utt
        y = T.tensor3() if out_every_t else T.matrix()
    return x, y        



def IRNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = theano.shared(np.identity(n_hidden, dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))

    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        if loss_function == 'CE':
            data_lin_output = V[x_t]
        else:
            data_lin_output = T.dot(x_t, V)
        
        h_t = T.nnet.relu(T.dot(h_prev, W) + data_lin_output + hidden_bias.dimshuffle('x', 0))
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        return h_t, cost_t, acc_t
    
    non_sequences = [V, W, hidden_bias, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info = [h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info = outputs_info)
   
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return inputs, parameters, costs



def tanhRNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = initialize_matrix(n_hidden, n_hidden, 'W', rng)
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))
    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        if loss_function == 'CE':
            data_lin_output = V[x_t]
        else:
            data_lin_output = T.dot(x_t, V)
        
        h_t = T.tanh(T.dot(h_prev, W) + data_lin_output + hidden_bias.dimshuffle('x', 0))
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        return h_t, cost_t, acc_t 
    
    non_sequences = [V, W, hidden_bias, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info = [h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
        
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)
   
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return inputs, parameters, costs



def LSTM(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE',flag_use_mask=False,flag_return_lin_output=False,flag_return_hidden_states=False,cost_weight=None,cost_transform=None,seed=1234):
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    W_i = initialize_matrix(n_input, n_hidden, 'W_i', rng)
    W_f = initialize_matrix(n_input, n_hidden, 'W_f', rng)
    W_c = initialize_matrix(n_input, n_hidden, 'W_c', rng)
    W_o = initialize_matrix(n_input, n_hidden, 'W_o', rng)
    U_i = initialize_matrix(n_hidden, n_hidden, 'U_i', rng)
    U_f = initialize_matrix(n_hidden, n_hidden, 'U_f', rng)
    U_c = initialize_matrix(n_hidden, n_hidden, 'U_c', rng)
    U_o = initialize_matrix(n_hidden, n_hidden, 'U_o', rng)
    V_o = initialize_matrix(n_hidden, n_hidden, 'V_o', rng)
    b_i = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    b_f = theano.shared(np.ones((n_hidden,), dtype=theano.config.floatX))
    b_c = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    b_o = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    state_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))
    parameters = [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, h_0, state_0, out_mat, out_bias]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    if flag_use_mask:
        if loss_function == 'CE':
            ymask = T.matrix(dtype='int8') if out_every_t else T.vector(dtype='int8')
        else:
            # y will be n_fram x n_output x n_utt
            ymask = T.tensor3(dtype='int8') if out_every_t else T.matrix(dtype='int8')
    
    def recurrence(x_t, y_t, ymask_t, h_prev, state_prev, cost_prev, acc_prev,
                   W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_mat, out_bias):
        
        if (loss_function == 'CE') and (input_type=='categorical'):
            x_t_W_i = W_i[x_t]
            x_t_W_c = W_c[x_t]
            x_t_W_f = W_f[x_t]
            x_t_W_o = W_o[x_t]
        else:
            x_t_W_i = T.dot(x_t, W_i)
            x_t_W_c = T.dot(x_t, W_c)
            x_t_W_f = T.dot(x_t, W_f)
            x_t_W_o = T.dot(x_t, W_o)
            
        input_t = T.nnet.sigmoid(x_t_W_i + T.dot(h_prev, U_i) + b_i.dimshuffle('x', 0))
        candidate_t = T.tanh(x_t_W_c + T.dot(h_prev, U_c) + b_c.dimshuffle('x', 0))
        forget_t = T.nnet.sigmoid(x_t_W_f + T.dot(h_prev, U_f) + b_f.dimshuffle('x', 0))

        state_t = input_t * candidate_t + forget_t * state_prev

        output_t = T.nnet.sigmoid(x_t_W_o + T.dot(h_prev, U_o) + T.dot(state_t, V_o) + b_o.dimshuffle('x', 0))

        h_t = output_t * T.tanh(state_t)

        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            if flag_use_mask:
                cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t, ymask_t=ymask_t)
            else:
                cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        return h_t, state_t, cost_t, acc_t

    non_sequences = [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    state_0_batch = T.tile(state_0, [x.shape[1], 1])
    
    if out_every_t:
        if flag_use_mask:
            sequences = [x, y, ymask]
        else:
            sequences = [x, y, T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    else:
        if flag_use_mask:
            sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1]), T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
        else:
            sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1]), T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)),[x.shape[0], 1, 1])]


    outputs_info = [h_0_batch, state_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
        
    [hidden_states, states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                          sequences=sequences,
                                                                          non_sequences=non_sequences,
                                                                          outputs_info=outputs_info)
    
    if flag_return_lin_output:
        #if output_type=='complex':
        #    lin_output = T.dot(hidden_states, out_mat) + out_bias.dimshuffle('x',0)
        #elif output_type=='real':
        lin_output = T.dot(hidden_states, out_mat) + out_bias.dimshuffle('x',0)
    
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)           
        cost=costs[0]
        accuracy=costs[1]
    else:
        if (cost_transform=='magTimesPhase'):
            cosPhase=T.cos(lin_output)
            sinPhase=T.sin(lin_output)
            linMag=np.sqrt(10**(x/10.0)-1e-5)
            yest_real=linMag*cosPhase
            yest_imag=linMag*sinPhase
            yest=T.concatenate([yest_real,yest_imag],axis=2)
            mse=(yest-y)**2
            cost_steps=T.mean(mse*ymask[:,:,0].dimshuffle(0,1,'x'),axis=2)
        elif cost_transform is not None:
            # assume that cost_transform is an inverse DFT followed by synthesis windowing
            lin_output_real=lin_output[:,:,:n_output//2]
            lin_output_imag=lin_output[:,:,n_output//2:]
            lin_output_sym_real=T.concatenate([lin_output_real,lin_output_real[:,:,n_output//2-2:0:-1]],axis=2)
            lin_output_sym_imag=T.concatenate([-lin_output_imag,lin_output_imag[:,:,n_output//2-2:0:-1]],axis=2)
            lin_output_sym=T.concatenate([lin_output_sym_real,lin_output_sym_imag],axis=2)
            yest_xform=T.dot(lin_output_sym,cost_transform)
            # apply synthesis window
            yest_xform=yest_xform*cost_weight.dimshuffle('x','x',0)
            y_real=y[:,:,:n_output//2]
            y_imag=y[:,:,n_output//2:]
            y_sym_real=T.concatenate([y_real,y_real[:,:,n_output//2-2:0:-1]],axis=2)
            y_sym_imag=T.concatenate([-y_imag,y_imag[:,:,n_output//2-2:0:-1]],axis=2)
            y_sym=T.concatenate([y_sym_real,y_sym_imag],axis=2)
            y_xform=T.dot(y_sym,cost_transform)
            # apply synthesis window
            y_xform=y_xform*cost_weight.dimshuffle('x','x',0)
            mse=(y_xform-yest_xform)**2
            cost_steps=T.mean(mse*ymask[:,:,0].dimshuffle(0,1,'x'),axis=2)
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

        if (loss_function=='CE_of_sum'):
            yest = T.sum(lin_output,axis=0) #sum over time_steps, yest is Nseq x n_output
            yest_softmax = T.nnet.softmax(yest)
            cost = T.nnet.categorical_crossentropy(yest_softmax, y[0,:]).mean()
            accuracy = T.eq(T.argmax(yest, axis=-1), y[0,:]).mean(dtype=theano.config.floatX)
            costs = [cost,accuracy]

    if flag_return_lin_output:
        costs = [cost, accuracy, lin_output]

        if flag_return_hidden_states:
            costs = costs + [hidden_states]

        #nmse_local = ymask.dimshuffle(0,1)*( (lin_output-y)**2 )/( 1e-5 + y**2 )
        nmse_local = theano.shared(np.float32(0.0))
        costs = costs + [nmse_local]
        
        costs = costs + [cost_steps]
    
    if flag_use_mask:
        return [x, y, ymask], parameters, costs
    else:
        return [x, y], parameters, costs

def initialize_unitary(n,impl,rng,name_suffix='',n_Givens=0,init='rand'):

    if (impl == 'adhoc'):
        # ad-hoc parameterization of Arjovsky, Shah, and Bengio 2015
        reflection = initialize_matrix(2, 2*n, 'reflection'+name_suffix, rng)
        theta = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                     high=np.pi,
                                                     size=(3, n)),
                                         dtype=theano.config.floatX), 
                              name='theta'+name_suffix)

        index_permute = rng.permutation(n)
        index_permute_long = np.concatenate((index_permute, index_permute + n))

        Wparams = [theta,reflection,index_permute_long]
    elif (impl == 'full'):
        """
        # fixed full unitary matrix
        Z=rng.randn(n,n).astype(np.complex64)+1j*rng.randn(n,n).astype(np.complex64)
        UZ, SZ, VZ=np.linalg.svd(Z)
        Wc=np.dot(UZ,VZ)
        WcRe=np.transpose(np.real(Wc))
        WcIm=np.transpose(np.imag(Wc))
        Waug = theano.shared(np.concatenate( [np.concatenate([WcRe,WcIm],axis=1),np.concatenate([(-1)*WcIm,WcRe],axis=1)], axis=0),name='Waug'+name_suffix)
        """
        if (init=='rand'):
            # use ad-hoc for initialization
            reflection = initialize_matrix(2, 2*n, 'reflection'+name_suffix, rng)
            theta = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                         high=np.pi,
                                                         size=(3, n)),
                                             dtype=theano.config.floatX), 
                                  name='theta'+name_suffix)

            index_permute = rng.permutation(n)
            index_permute_long = np.concatenate((index_permute, index_permute + n))

            WcRe=np.eye(n).astype(np.float32)
            WcIm=np.zeros((n,n)).astype(np.float32)
            Waug=np.concatenate( [np.concatenate([WcRe,WcIm],axis=1),np.concatenate([WcIm,WcRe],axis=1)], axis=0)
            swap_re_im = np.concatenate((np.arange(n, 2*n), np.arange(n)))
            Waug_variable=times_unitary(Waug,n,swap_re_im,[theta,reflection,index_permute_long],'adhoc')
            Waug=theano.shared(Waug_variable.eval().astype(np.float32),name='Waug'+name_suffix)
        elif (init=='identity'):
            WcRe=np.eye(n).astype(np.float32)
            WcIm=np.zeros((n,n)).astype(np.float32)
            Waug_np=np.concatenate( [np.concatenate([WcRe,WcIm],axis=1),np.concatenate([WcIm,WcRe],axis=1)], axis=0)
            Waug=theano.shared(Waug_np,name='Waug'+name_suffix)
        Wparams = [Waug]
    elif (impl == 'givens'):
        # composition of Givens rotations
        gphipsi = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                       high=np.pi,
                                                       size=(n_Givens, 1, 2)),
                                           dtype=theano.config.floatX), 
                                name='gphipsi') 
        gchi = theano.shared(np.asarray(np.arccos(rng.uniform(low=0,
                                                   high=1,
                                                   size=(n_Givens, 1, 1))),
                                        dtype=theano.config.floatX), 
                             name='gchi')
        #galp = theano.shared(np.asarray(rng.uniform(low=-np.pi,
        #                                            high=np.pi,
        #                                            size=(1, 1)),
        #                                dtype=theano.config.floatX),
        #                     name='galp')

        # build indices for Givens rotations:
        Nchoose2=(n)*(n-1)/2;
        #  generate a random permutation of 1:(N choose 2)
        Nchoose2perm=rng.permutation(Nchoose2)
        # take the first n_Givens values
        Nchoose2perm=Nchoose2perm[:n_Givens]
        # initialize Givens indices
        gidx_np=np.zeros((n_Givens,1,2),dtype=np.int32)
        ig=0 #absolute Givens index
        ig_sel=0 #indices for selected Givens indices
        for ig1 in range(0,n):
            for ig2 in range(ig1+1,n):
                if ig in Nchoose2perm:
                    ig_sel=np.where(Nchoose2perm==ig)
                    ig_sel=ig_sel[0][0]
                    gidx_np[ig_sel,0,:]=np.reshape([ig1,ig2],(1,1,2))
                ig=ig+1
        gidx=theano.shared(gidx_np)
        Wparams = [gphipsi, gchi, gidx]

    return Wparams

def initialize_complex_RNN_layer(n_hidden,Wimpl,rng,hidden_bias_mean,name_suffix='',n_Givens=0,hidden_bias_init='rand',h_0_init='rand',W_init='rand'):
    # hidden bias
    if (hidden_bias_init=='rand'):
        hidden_bias = theano.shared(np.asarray(hidden_bias_mean+rng.uniform(low=-0.01,
                                                           high=0.01,
                                                           size=(n_hidden,)),
                                               dtype=theano.config.floatX), 
                                    name='hidden_bias'+name_suffix)
    elif (hidden_bias_init=='zero'):
        hidden_bias = theano.shared(np.zeros((n_hidden,)).astype(theano.config.floatX),name='hidden_bias'+name_suffix)
    else:
        raise ValueError("Unknown initialization method %s for hidden_bias" % hidden_bias_init)

    # initial state h_0
    h_0_size=(1,2*n_hidden)
    if (h_0_init=='rand'):
        bucket = np.sqrt(3. / 2 / n_hidden) 
        h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                                   high=bucket,
                                                   size=h_0_size), 
                                       dtype=theano.config.floatX),
                            name='h_0'+name_suffix)
    elif (h_0_init=='zero'):
        h_0 = theano.shared(np.zeros(h_0_size).astype(theano.config.floatX),name='h_0'+name_suffix)
    else:
        raise ValueError("Unknown initialization method %s for h_0" % h_0_init)

    # unitary transition matrix W
    Wparams = initialize_unitary(n_hidden,Wimpl,rng,name_suffix=name_suffix,n_Givens=n_Givens,init=W_init)
    """
    if (Wimpl == 'adhoc'):
        # ad-hoc parameterization of Arjovsky, Shah, and Bengio 2015
        reflection = initialize_matrix(2, 2*n_hidden, 'reflection'+name_suffix, rng)
        theta = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                     high=np.pi,
                                                     size=(3, n_hidden)),
                                         dtype=theano.config.floatX), 
                              name='theta'+name_suffix)

        index_permute = np.random.permutation(n_hidden)
        index_permute_long = np.concatenate((index_permute, index_permute + n_hidden))

        Wparams = [theta,reflection,index_permute_long]
    elif (Wimpl == 'full'):
        # fixed full unitary matrix
        Z=prng_Givens.randn(n_hidden,n_hidden).astype(np.complex64)+1j*prng_Givens.randn(n_hidden,n_hidden).astype(np.complex64)
        UZ, SZ, VZ=np.linalg.svd(Z)
        Wc=np.dot(UZ,VZ)
        WcRe=np.transpose(np.real(Wc))
        WcIm=np.transpose(np.imag(Wc))
        Waug = theano.shared(np.concatenate( [np.concatenate([WcRe,WcIm],axis=1),np.concatenate([(-1)*WcIm,WcRe],axis=1)], axis=0),name='Waug'+name_suffix)

        Wparams = [Waug]
    elif (Wimpl == 'givens'):
        # composition of Givens rotations
        gphipsi = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                       high=np.pi,
                                                       size=(n_Givens, 1, 2)),
                                           dtype=theano.config.floatX), 
                                name='gphipsi') 
        gchi = theano.shared(np.asarray(np.arccos(rng.uniform(low=0,
                                                   high=1,
                                                   size=(n_Givens, 1, 1))),
                                        dtype=theano.config.floatX), 
                             name='gchi')
        #galp = theano.shared(np.asarray(rng.uniform(low=-np.pi,
        #                                            high=np.pi,
        #                                            size=(1, 1)),
        #                                dtype=theano.config.floatX),
        #                     name='galp')

        # build indices for Givens rotations:
        Nchoose2=(n_hidden)*(n_hidden-1)/2;
        #  generate a random permutation of 1:(N choose 2)
        Nchoose2perm=prng_Givens.permutation(Nchoose2)
        # take the first n_Givens values
        Nchoose2perm=Nchoose2perm[:n_Givens]
        # initialize Givens indices
        gidx_np=np.zeros((n_Givens,1,2),dtype=np.int32)
        ig=0 #absolute Givens index
        ig_sel=0 #indices for selected Givens indices
        for ig1 in range(0,n_hidden):
            for ig2 in range(ig1+1,n_hidden):
                if ig in Nchoose2perm:
                    ig_sel=np.where(Nchoose2perm==ig)
                    ig_sel=ig_sel[0][0]
                    gidx_np[ig_sel,0,:]=np.reshape([ig1,ig2],(1,1,2))
                ig=ig+1
        gidx=theano.shared(gidx_np)
        Wparams = [gphipsi, gchi, gidx]
    """

    return hidden_bias, h_0, Wparams

def times_unitary(x,n,swap_re_im,Wparams,Wimpl):
    # multiply tensor x on the right  by the unitary matrix W parameterized by Wparams
    if (Wimpl == 'adhoc'):
        theta=Wparams[0]
        reflection=Wparams[1]
        index_permute_long=Wparams[2]
        step1 = times_diag(x, n, theta[0,:], swap_re_im)
        step2 = do_fft(step1, n)
        step3 = times_reflection(step2, n, reflection[0,:])
        step4 = vec_permutation(step3, index_permute_long)
        step5 = times_diag(step4, n, theta[1,:], swap_re_im)
        step6 = do_ifft(step5, n)
        step7 = times_reflection(step6, n, reflection[1,:])
        step8 = times_diag(step7, n, theta[2,:], swap_re_im)     
        y = step8
    elif (Wimpl == 'full'):
        Waug=Wparams[0]
        y = T.dot(x,Waug)
    elif (Wimpl == 'givens'):
        gphipsi=Wparams[0]
        gchi=Wparams[1]
        gidx=Wparams[2]
        # scan method for composing Givens rotations
        givens_steps=h_prev #output of this inner scan should be I x 2*n
        givens_outputs, updates = theano.scan(fn=lambda gphipsi, 
                                                        gchi, 
                                                        gidx, 
                                                        Gh_prev: 
                                                        times_givens(Gh_prev,
                                                                     n,
                                                                     T.reshape(T.concatenate([gphipsi,gchi],axis=1),[3]),
                                                                     T.reshape(gidx,[2])),
                                              sequences=[gphipsi,gchi,gidx],
                                              outputs_info=givens_steps)
        # output of composition of Givens rotations:
        y=T.reshape(givens_outputs[-1,:,:],(givens_outputs.shape[1],givens_outputs.shape[2]))
    return y


def complex_RNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE', output_type='real', fidx=None, flag_return_lin_output=False,name_suffix='',x_spec=None,flag_feed_forward=False,flag_use_mask=False,hidden_bias_mean=0.0,lam=0.0,Wimpl="adhoc",n_Givens=None,prng_Givens=np.random.RandomState(),Vnorm=0.0,Unorm=0.0,flag_return_hidden_states=False,n_layers=1,cost_weight=None,cost_transform=None,flag_noComplexConstraint=0,seed=1234,V_init='rand',U_init='rand',W_init='rand',h_0_init='rand',out_bias_init='rand',hidden_bias_init='rand',flag_add_input_to_output=False,flag_connect_input_to_layers=False,flag_broadcast_silo=False):

    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Initialize input and output parameters: V, U, out_bias0
    
    # input matrix V
    if flag_noComplexConstraint and (input_type=='complex'):
        V = initialize_matrix(2*n_input, 2*n_hidden, 'V'+name_suffix, rng, init=V_init)
        Vaug = V
    else:
        V = initialize_matrix(n_input, 2*n_hidden, 'V'+name_suffix, rng, init=V_init)
        if (Vnorm>0.0):
            # normalize the rows of V by the L2 norm (note that the variable V here is actually V^T, so we normalize the columns)
            Vr = V[:,:n_hidden]
            Vi = V[:,n_hidden:]
            Vnorms = T.sqrt(1e-5 + T.sum(Vr**2,axis=0,keepdims=True) + T.sum(Vi**2,axis=0,keepdims=True))
            Vn = T.concatenate( [Vr/(1e-5 + Vnorms), Vi/(1e-5 + Vnorms)], axis=1)
            # scale so row norms are desired number
            Vn = V*T.sqrt(Vnorm)
        else:
            Vn = V

        if input_type=='complex':
            Vim = T.concatenate([ (-1)*Vn[:,n_hidden:], Vn[:,:n_hidden] ],axis=1) #concatenate along columns to make [-V_I, V_R]
            Vaug = T.concatenate([ Vn, Vim ],axis=0) #concatenate along rows to make [V_R, V_I; -V_I, V_R]
    

    # output matrix U
    if flag_noComplexConstraint and (input_type=='complex'):
        U = initialize_matrix(2*n_hidden,2*n_output,'U'+name_suffix,rng, init=U_init)
        Uaug=U
    else:
        U = initialize_matrix(2 * n_hidden, n_output, 'U'+name_suffix, rng, init=U_init)
        if (Unorm > 0.0):
            # normalize the cols of U by the L2 norm (note that the variable U here is actually U^H, so we normalize the rows)
            Ur = U[:n_hidden,:]
            Ui = U[n_hidden:,:]
            Unorms = T.sqrt(1e-5 + T.sum(Ur**2,axis=1,keepdims=True) + T.sum(Ui**2,axis=1,keepdims=True))
            Un = T.concatenate([ Ur/(1e-5 + Unorms), Ui/(1e-5 + Unorms) ], axis=0)
            # scale so col norms are desired number
            Un = Un*T.sqrt(Unorm)
        else:
            Un = U

        if output_type=='complex':
            Uim = T.concatenate([ (-1)*Un[n_hidden:,:], Un[:n_hidden,:] ],axis=0) #concatenate along rows to make [-U_I; U_R]
            Uaug = T.concatenate([ Un,Uim ],axis=1) #concatante along cols to make [U_R, -U_I; U_I, U_R]
            # note that this is a little weird compared to the convention elsewhere in this code that
            # right-multiplication real-composite form is [A, B; -B, A]. The weirdness is because of the original
            # implementation, which initialized U for real-valued outputs as U=[A; B], which really should have
            # been U=[A; -B]

    
    # output bias out_bias
    if output_type=='complex':
        out_bias = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX), name='out_bias'+name_suffix)
    else:
        out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias'+name_suffix)
   
    
    # initialize layer 1 parameters
    hidden_bias, h_0, Wparams = initialize_complex_RNN_layer(n_hidden,Wimpl,rng,hidden_bias_mean,name_suffix=name_suffix,n_Givens=n_Givens,hidden_bias_init=hidden_bias_init,h_0_init=h_0_init,W_init=W_init)

    # extract recurrent parameters into this namespace 
    if flag_feed_forward:
        # just doing feed-foward, so remove any recurrent parameters
        if (Wimpl == 'adhoc'):
            #theta = theano.shared(np.float32(0.0))
            h_0_size=(1,2*n_hidden)
            h_0 = theano.shared(np.asarray(np.zeros(h_0_size),dtype=theano.config.floatX))
        
        parameters = [V, U, hidden_bias, out_bias]
       
    else:
        if (Wimpl == 'adhoc'):
            # ad-hoc parameterization of Arjovsky, Shah, and Bengio 2015
            theta = Wparams[0]
            reflection = Wparams[1]
            index_permute_long = Wparams[2]

            parameters = [V, U, hidden_bias, reflection, out_bias, theta, h_0]
            #Wparams = [theta]
        elif (Wimpl == 'full'):
            # fixed full unitary matrix
            Waug=Wparams[0]

            parameters = [V, U, hidden_bias, out_bias, h_0, Waug]
            #Wparams = [Waug]
        elif (Wimpl == 'givens'):
            # composition of Givens rotations
            gphipsi = Wparams[0]
            gchi    = Wparams[1]
            gidx    = Wparams[2]
            
            parameters = [V,U, hidden_bias, out_bias, h_0, gphipsi, gchi]
            #Wparams = [gphipsi, gchi, gidx]

    h_0_all_layers = h_0

    # initialize additional layer parameters
    addl_layers_params=[]
    addl_layers_params_optim=[]
    for i_layer in range(2,n_layers+1):
        betw_layer_suffix='_L%d_to_L%d' % (i_layer-1,i_layer)
        layer_suffix='_L%d' % i_layer
        Wvparams_cur = initialize_unitary(n_hidden,Wimpl,rng,name_suffix=(name_suffix+betw_layer_suffix),n_Givens=n_Givens,init=W_init)
        hidden_bias_cur, h_0_cur, Wparams_cur = initialize_complex_RNN_layer(n_hidden,Wimpl,rng,hidden_bias_mean,name_suffix=(name_suffix + layer_suffix),n_Givens=n_Givens,hidden_bias_init=hidden_bias_init,h_0_init=h_0_init,W_init=W_init)
                
        addl_layers_params = addl_layers_params + Wvparams_cur + [hidden_bias_cur, h_0_cur, ] + Wparams_cur
        if (Wimpl=='adhoc'):
            # don't include permutation indices in the list of parameters to be optimized
            addl_layers_params_optim = addl_layers_params_optim + Wvparams_cur[0:2] + [hidden_bias_cur, h_0_cur] + Wparams_cur[0:2]
        else:
            addl_layers_params_optim = addl_layers_params
        
        if flag_connect_input_to_layers:
            Vk = initialize_matrix(n_input, 2*n_hidden, 'V'+name_suffix+layer_suffix, rng, init=V_init)
            if (Vnorm>0.0):
                # normalize the rows of V by the L2 norm (note that the variable V here is actually V^T, so we normalize the columns)
                Vkr = Vk[:,:n_hidden]
                Vki = Vk[:,n_hidden:]
                Vknorms = T.sqrt(1e-5 + T.sum(Vkr**2,axis=0,keepdims=True) + T.sum(Vki**2,axis=0,keepdims=True))
                Vkn = T.concatenate( [Vkr/(1e-5 + Vknorms), Vki/(1e-5 + Vknorms)], axis=1)
                # scale so row norms are desired number
                Vkn = Vk*T.sqrt(Vknorm)
            else:
                Vkn = Vk

            if input_type=='complex':
                Vkim = T.concatenate([ (-1)*Vkn[:,n_hidden:], Vkn[:,:n_hidden] ],axis=1) #concatenate along columns to make [-V_I, V_R]
                Vkaug = T.concatenate([ Vkn, Vkim ],axis=0) #concatenate along rows to make [V_R, V_I; -V_I, V_R]
                addl_layers_params = addl_layers_params + [Vkaug]
            else:
                addl_layers_params = addl_layers_params + [Vkn]
            addl_layers_params_optim = addl_layers_params_optim + [Vk]

        h_0_all_layers = T.concatenate([h_0_all_layers,h_0_cur],axis=1)

    parameters = parameters + addl_layers_params_optim

    # initialize data nodes
    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    if flag_use_mask:
        if 'CE' in loss_function:
            ymask = T.matrix(dtype='int8') if out_every_t else T.vector(dtype='int8')
        else:
            # y will be n_fram x n_output x n_utt
            ymask = T.tensor3(dtype='int8') if out_every_t else T.matrix(dtype='int8')

    if x_spec is not None:
        # x is specified, set x to this:
        x = x_spec
    
    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))


    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, ymask_t, h_prev, cost_prev, acc_prev, V, hidden_bias, out_bias, U, *argv):  

        # h_prev is of size n_batch x n_layers*2*n_hidden

        # strip W parameters off variable arguments list
        if (Wimpl=='full'):
            Wparams=argv[0:1]
            argv=argv[1:]
        else:
            Wparams=argv[0:3]
            argv=argv[3:]

        if not flag_feed_forward:
            # Compute hidden linear transform: W h_{t-1}
            h_prev_layer1 = h_prev[:,0:2*n_hidden]
            hidden_lin_output = times_unitary(h_prev_layer1,n_hidden,swap_re_im,Wparams,Wimpl)

        # Compute data linear transform
        if ('CE' in loss_function) and (input_type=='categorical'):
            # inputs are categorical, so just use them as indices into V
            data_lin_output = V[T.cast(x_t, 'int32')]
        else:
            # second dimension of real-valued x_t should be of size n_input, first dimension of V should be of size n_input
            # (or augmented, where the dimension of summation is 2*n_input and V is of real/imag. augmented form)
            data_lin_output = T.dot(x_t, V)
            
        # Total linear output        
        if not flag_feed_forward:
            lin_output = hidden_lin_output + data_lin_output
        else:
            lin_output = data_lin_output

        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        #  add a little bit to sqrt argument to ensure stable gradients,
        #  since gradient of sqrt(x) is -0.5/sqrt(x)
        modulus = T.sqrt(1e-9+lin_output**2 + lin_output[:, swap_re_im]**2)
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus)
        h_t = lin_output * rescale
     
        h_t_all_layers = h_t

        # Compute additional recurrent layers
        for i_layer in range(2,n_layers+1):
            
            # strip Wv parameters off variable arguments list
            if (Wimpl=='full'):
                Wvparams_cur=argv[0:1]
                argv=argv[1:]
            else:
                Wvparams_cur=argv[0:3]
                argv=argv[3:]
            
            # strip hidden_bias for this layer off argv
            hidden_bias_cur = argv[0]
            argv=argv[1:]
            
            # strip h_0 for this layer off argv
            #h_0_cur = argv[0] #unused, since h_0_all_layers is all layers' h_0s concatenated
            argv=argv[1:]
            
            # strip W parameters off variable arguments list
            if (Wimpl=='full'):
                Wparams_cur=argv[0:1]
                argv=argv[1:]
            else:
                Wparams_cur=argv[0:3]
                argv=argv[3:]

            if flag_connect_input_to_layers:
                # strip layer-dependent input transforms off variable arugments list
                Vk=argv[0]
                argv=argv[1:]

            # Compute the linear parts of the layer ----------

            if not flag_feed_forward:
                # get previous hidden state h_{t-1} for this layer:
                if flag_broadcast_silo:
                    # use top of the previous iteration stack for h_{t-1}
                    h_prev_cur = h_prev[:,(n_layers-1)*2*n_hidden:n_layers*2*n_hidden]
                else:
                    h_prev_cur = h_prev[:,(i_layer-1)*2*n_hidden:i_layer*2*n_hidden]
                # Compute hidden linear transform: W h_{t-1}
                hidden_lin_output_cur = times_unitary(h_prev_cur,n_hidden,swap_re_im,Wparams_cur,Wimpl)

            # Compute "data linear transform", which for this intermediate layer is the previous layer's h_t transformed by Wv
            data_lin_output_cur = times_unitary(h_t,n_hidden,swap_re_im,Wvparams_cur,Wimpl)
                
            # Total linear output        
            if not flag_feed_forward:
                lin_output_cur = hidden_lin_output_cur + data_lin_output_cur
            else:
                lin_output_cur = data_lin_output_cur

            if flag_connect_input_to_layers:
                lin_output_cur = lin_output_cur + T.dot(x_t,Vk)

            # Apply non-linearity ----------------------------

            # scale RELU nonlinearity
            #  add a little bit to sqrt argument to ensure stable gradients,
            #  since gradient of sqrt(x) is -0.5/sqrt(x)
            modulus = T.sqrt(1e-9+lin_output_cur**2 + lin_output_cur[:, swap_re_im]**2)
            rescale = T.maximum(modulus + T.tile(hidden_bias_cur, [2]).dimshuffle('x', 0), 0.) / (modulus)
            h_t = lin_output_cur * rescale
            h_t_all_layers = T.concatenate([h_t_all_layers,h_t],axis=1)

        # assume we aren't passing any preactivation to compute_cost
        z_t = None

        if loss_function == 'MSEplusL1':
            z_t = h_t

        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
    
            if flag_add_input_to_output:
                lin_output=lin_output + x_t 

            if flag_use_mask:
                cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t, ymask_t=ymask_t, z_t=z_t, lam=lam)
            else:
                cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t, z_t=z_t, lam=lam)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t_all_layers, cost_t, acc_t

    def recurrence_Givens(x_t, y_t, ymask_t, h_prev, cost_prev, acc_prev, V, hidden_bias, out_bias, U, gphipsi, gchi, gidx):  

        if not flag_feed_forward:
            # scan method for composing Givens rotations
            givens_steps=h_prev #output of this inner scan should be I x 2*n_hidden
            givens_outputs, updates = theano.scan(fn=lambda gphipsi, 
                                                            gchi, 
                                                            gidx, 
                                                            Gh_prev: 
                                                            times_givens(Gh_prev,
                                                                         n_hidden,
                                                                         T.reshape(T.concatenate([gphipsi,gchi],axis=1),[3]),
                                                                         T.reshape(gidx,[2])),
                                                  sequences=[gphipsi,gchi,gidx],
                                                  outputs_info=givens_steps)
            # output of composition of Givens rotations:
            hidden_lin_output=T.reshape(givens_outputs[-1,:,:],(givens_outputs.shape[1],givens_outputs.shape[2]))

        # Compute data linear transform
        if ('CE' in loss_function) and (input_type=='categorical'):
            # inputs are categorical, so just use them as indices into V
            data_lin_output = V[T.cast(x_t, 'int32')]
        else:
            # second dimension of real-valued x_t should be of size n_input, first dimension of V should be of size n_input
            # (or augmented, where the dimension of summation is 2*n_input and V is of real/imag. augmented form)
            data_lin_output = T.dot(x_t, V)
            
        # Total linear output        
        if not flag_feed_forward:
            lin_output = hidden_lin_output + data_lin_output
        else:
            lin_output = data_lin_output


        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        #  add a little bit to sqrt argument to ensure stable gradients,
        #  since gradient of sqrt(x) is -0.5/sqrt(x)
        modulus = T.sqrt(1e-9+lin_output**2 + lin_output[:, swap_re_im]**2)
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus)
        h_t = lin_output * rescale
       
        # assume we aren't passing any preactivation to compute_cost
        z_t = None

        if loss_function == 'MSEplusL1':
            z_t = h_t

        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            if flag_use_mask:
                cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t, ymask_t=ymask_t, z_t=z_t, lam=lam)
            else:
                cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t, z_t=z_t, lam=lam)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t, cost_t, acc_t
    
    # compute hidden states
    #  h_0_batch should be n_utt x n_layers*2*n_hidden, since scan goes over first dimension of x, which is the maximum STFT length in frames
    h_0_batch = T.tile(h_0_all_layers, [x.shape[1], 1])
    if (Wimpl=='givens'):
        if input_type=='complex' and output_type=='complex':
            # pass in augmented input and output transformations
            non_sequences = [Vaug, hidden_bias, out_bias, Uaug, gphipsi, gchi, gidx]
        elif input_type=='complex':
            non_sequences = [Vaug, hidden_bias, out_bias, Un, gphipsi, gchi, gidx]
        elif output_type=='complex':
            non_sequences = [Vn   , hidden_bias, out_bias, Uaug, gphipsi, gchi, gidx]
        else:
            non_sequences = [Vn   , hidden_bias, out_bias, Un, gphipsi, gchi, gidx]
    else:
        if input_type=='complex' and output_type=='complex':
            # pass in augmented input and output transformations
            non_sequences = [Vaug, hidden_bias, out_bias, Uaug] + Wparams + addl_layers_params
        elif input_type=='complex':
            non_sequences = [Vaug, hidden_bias, out_bias, Un] + Wparams + addl_layers_params
        elif output_type=='complex':
            non_sequences = [Vn   , hidden_bias, out_bias, Uaug] + Wparams + addl_layers_params
        else:
            non_sequences = [Vn   , hidden_bias, out_bias, Un] + Wparams + addl_layers_params
    
    if out_every_t:
        if flag_use_mask:
            sequences = [x, y, ymask]
        else:
            sequences = [x, y, T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    else:
        if flag_use_mask:
            sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1]), T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
        else:
            sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1]), T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)),[x.shape[0], 1, 1])]

    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    if (Wimpl=='givens'):
        [hidden_states_all_layers, cost_steps, acc_steps], updates = theano.scan(fn=recurrence_Givens,
                                                                      sequences=sequences,
                                                                      non_sequences=non_sequences,
                                                                      outputs_info=outputs_info)
    else:
        [hidden_states_all_layers, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                      sequences=sequences,
                                                                      non_sequences=non_sequences,
                                                                      outputs_info=outputs_info)

    # get hidden states of last layer
    hidden_states = hidden_states_all_layers[:,:,(n_layers-1)*2*n_hidden:]

    if flag_return_lin_output:
        if output_type=='complex':
            lin_output = T.dot(hidden_states, Uaug) + out_bias.dimshuffle('x',0)
        else:
            lin_output = T.dot(hidden_states, Un) + out_bias.dimshuffle('x',0)
   
        if flag_add_input_to_output:
            lin_output = lin_output + x

    if not out_every_t:
        #TODO: here, if flag_use_mask is set, need to use a for-loop to select the desired time-step for each utterance
        lin_output = T.dot(hidden_states[-1,:,:], Un) + out_bias.dimshuffle('x', 0)
        z_t = None
        if loss_function == 'MSEplusL1':
            z_t = hidden_states[-1,:,:]
        costs = compute_cost_t(lin_output, loss_function, y, z_t=z_t, lam=lam)
        cost=costs[0]
        accuracy=costs[1]
    else:
        if (cost_transform=='magTimesPhase'):
            cosPhase=T.cos(lin_output)
            sinPhase=T.sin(lin_output)
            linMag=np.sqrt(10**(x/10.0)-1e-5)
            yest_real=linMag*cosPhase
            yest_imag=linMag*sinPhase
            yest=T.concatenate([yest_real,yest_imag],axis=2)
            mse=(yest-y)**2
            cost_steps=T.mean(mse*ymask[:,:,0].dimshuffle(0,1,'x'),axis=2)
        elif cost_transform is not None:
            # assume that cost_transform is an inverse DFT followed by synthesis windowing
            lin_output_real=lin_output[:,:,:n_output]
            lin_output_imag=lin_output[:,:,n_output:]
            lin_output_sym_real=T.concatenate([lin_output_real,lin_output_real[:,:,n_output-2:0:-1]],axis=2)
            lin_output_sym_imag=T.concatenate([-lin_output_imag,lin_output_imag[:,:,n_output-2:0:-1]],axis=2)
            lin_output_sym=T.concatenate([lin_output_sym_real,lin_output_sym_imag],axis=2)
            yest_xform=T.dot(lin_output_sym,cost_transform)
            # apply synthesis window
            yest_xform=yest_xform*cost_weight.dimshuffle('x','x',0)
            y_real=y[:,:,:n_output]
            y_imag=y[:,:,n_output:]
            y_sym_real=T.concatenate([y_real,y_real[:,:,n_output-2:0:-1]],axis=2)
            y_sym_imag=T.concatenate([-y_imag,y_imag[:,:,n_output-2:0:-1]],axis=2)
            y_sym=T.concatenate([y_sym_real,y_sym_imag],axis=2)
            y_xform=T.dot(y_sym,cost_transform)
            # apply synthesis window
            y_xform=y_xform*cost_weight.dimshuffle('x','x',0)
            mse=(y_xform-yest_xform)**2
            cost_steps=T.mean(mse*ymask[:,:,0].dimshuffle(0,1,'x'),axis=2)
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
   
        if (loss_function=='CE_of_sum'):
            yest = T.sum(lin_output,axis=0) #sum over time_steps, yest is Nseq x n_output
            yest_softmax = T.nnet.softmax(yest)
            cost = T.nnet.categorical_crossentropy(yest_softmax, y[0,:]).mean()
            accuracy = T.eq(T.argmax(yest, axis=-1), y[0,:]).mean(dtype=theano.config.floatX)

    if flag_return_lin_output:

        costs = [cost, accuracy, lin_output]
        
        if flag_return_hidden_states:
            costs = costs + [hidden_states]

        #nmse_local = ymask.dimshuffle(0,1)*( (lin_output-y)**2 )/( 1e-5 + y**2 )
        nmse_local = theano.shared(np.float32(0.0))
        costs = costs + [nmse_local]

        costs = costs + [cost_steps]

    else:
        costs = [cost, accuracy]
    if flag_use_mask:
        return [x,y,ymask], parameters, costs
    else:
        return [x, y], parameters, costs


def Givens_RNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE', output_type='real', fidx=None, flag_return_lin_output=False, flag_useGivensForLoop=False):
    # composes {n_hidden}\choose{2} Givens rotations to form W, which is guaranteed
    # to cover the entire unitary group U(n_hidden) [K Zyczkowski, M Kus, "Random Unitary Matrices", 1994]

    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    if input_type=='complex':
        V = initialize_matrix(n_input, 2*n_hidden, 'V', rng)
        Vim = T.concatenate([ (-1)*V[:,n_hidden:], V[:,:n_hidden] ],axis=1) #concatenate along columns to make [-V_I, V_R]
        Vaug = T.concatenate([ V, Vim ],axis=0) #concatenate along rows to make [V_R, V_I; -V_I, V_R]
    else:
        V = initialize_matrix(n_input, 2*n_hidden, 'V', rng)
    if output_type=='complex':
        U = initialize_matrix(2 * n_hidden, n_output, 'U', rng)
        Uim = T.concatenate([ (-1)*U[n_hidden:,:], U[:n_hidden,:] ],axis=0) #concatenate along rows to make [-U_I; U_R]
        Uaug = T.concatenate([ U,Uim ],axis=1) #concatante along columns to make [U_R, U_I; -U_I, U_R]
    else:
        U = initialize_matrix(2 * n_hidden, n_output, 'U', rng)

    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias')
    
    if output_type=='complex':
        out_bias = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX), name='out_bias')
    else:
        out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')
    Nchoose2=(n_hidden)*(n_hidden-1)/2;
    
    gphipsi = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                   high=np.pi,
                                                   size=(Nchoose2, 1, 2)),
                                       dtype=theano.config.floatX), 
                            name='gphipsi') 
    gchi = theano.shared(np.asarray(np.arccos(rng.uniform(low=0,
                                               high=1,
                                               size=(Nchoose2, 1, 1))),
                                    dtype=theano.config.floatX), 
                         name='gchi')
    #galp = theano.shared(np.asarray(rng.uniform(low=-np.pi,
    #                                            high=np.pi,
    #                                            size=(1, 1)),
    #                                dtype=theano.config.floatX),
    #                     name='galp')

    # build indices for Givens rotations:
    gidx=np.zeros((Nchoose2,1,2),dtype=np.int32)
    ig=0
    for ig1 in range(0,n_hidden):
        for ig2 in range(ig1+1,n_hidden):
            gidx[ig,0,:]=np.reshape([ig1,ig2],(1,1,2))
            ig=ig+1

    bucket = np.sqrt(3. / 2 / n_hidden) 
    h_0_size=(1,2*n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=h_0_size), 
                                   dtype=theano.config.floatX),
                        name='h_0')
    
    parameters = [V, U, hidden_bias, out_bias, h_0, gphipsi, gchi]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    
    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))
    
    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, hidden_bias, out_bias, U, gphipsi, gchi, gidx):  

        # h_prev is nutt x 2*n_hidden

        # Compute hidden linear transform
        ##complex_RNN steps
        #step1 = times_diag(h_prev, n_hidden, theta[0,:], swap_re_im)
        #step2 = do_fft(step1, n_hidden)
        #step3 = times_reflection(step2, n_hidden, reflection[0,:])
        #step4 = vec_permutation(step3, index_permute_long)
        #step5 = times_diag(step4, n_hidden, theta[1,:], swap_re_im)
        #step6 = do_ifft(step5, n_hidden)
        #step7 = times_reflection(step6, n_hidden, reflection[1,:])
        #step8 = times_diag(step7, n_hidden, theta[2,:], swap_re_im)     
        #
        #hidden_lin_output = step8
 
        if flag_useGivensForLoop:
            # for loop method
            hidden_lin_output=h_prev
            ig=0; #absolute matrix index
            for ig1 in range(0,n_hidden):
                for ig2 in range(ig1+1,n_hidden):
                    hidden_lin_output=T.set_subtensor(hidden_lin_output[:,[ig1,ig2,ig1+n_hidden,ig2+n_hidden]],
                                                      times_givens(hidden_lin_output[:,[ig1,ig2,ig1+n_hidden,ig2+n_hidden]],
                                                                   2,
                                                                   T.reshape(T.concatenate([gphipsi[ig,:],gchi[ig,:]],axis=1),[3]),
                                                                   np.asarray([1,2],dtype=np.int32)))
        else:
            # scan method for composing Givens rotations
            givens_steps=h_prev #output of this inner scan should be I x 2*n_hidden
            givens_outputs, updates = theano.scan(fn=lambda gphipsi, 
                                                            gchi, 
                                                            gidx, 
                                                            Gh_prev: 
                                                            times_givens(Gh_prev,
                                                                         n_hidden,
                                                                         T.reshape(T.concatenate([gphipsi,gchi],axis=1),[3]),
                                                                         T.reshape(gidx,[2])),
                                                  sequences=[gphipsi,gchi,gidx],
                                                  outputs_info=[givens_steps])
            # output of composition of Givens rotations:
            hidden_lin_output=T.reshape(givens_outputs[-1,:,:],(givens_outputs.shape[1],givens_outputs.shape[2]))

        # Compute data linear transform
        if loss_function == 'CE':
            # inputs are categorical, so just use them as indices into V
            data_lin_output = V[T.cast(x_t, 'int32')]
        # elif input_type=='complex':
        #     # second dimension of complex-valued x_t should be of size 2*n_input, with 0:(n_input-1) the real part and
        #     # n_input:end the imag. part
        #     data_lin_output = T.dot(x_t, V)
        else:
            # second dimension of real-valued x_t should be of size n_input, first dimension of V should be of size n_input
            # (or augmented, where the dimension of summation is 2*n_input and V is of real/imag. augmented form)
            data_lin_output = T.dot(x_t, V)
            
        # Total linear output        
        lin_output = hidden_lin_output + data_lin_output


        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output**2 + lin_output[:, swap_re_im]**2)
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
        h_t = lin_output * rescale
        
        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t, cost_t, acc_t

    # compute hidden states
    #  h_0_batch should be n_utt x 2*n_hidden, since scan goes over first dimension of x, which is the maximum STFT length in frames
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    if input_type=='complex' and output_type=='complex':
        # pass in augmented input and output transformations
        non_sequences = [Vaug, hidden_bias, out_bias, Uaug, gphipsi, gchi, gidx]
    elif input_type=='complex':
        non_sequences = [Vaug, hidden_bias, out_bias, U, gphipsi, gchi, gidx]
    elif output_type=='complex':
        non_sequences = [V   , hidden_bias, out_bias, Uaug, gphipsi, gchi, gidx]
    else:
        non_sequences = [V, hidden_bias, out_bias, U, gphipsi, gchi, gidx]
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        if flag_return_lin_output:
            if output_type=='complex':
                lin_outputs = T.dot(hidden_states, Uaug) + out_bias.dimshuffle('x',0)
            elif output_type=='real':
                lin_outputs = T.dot(hidden_states, U) + out_bias.dimshuffle('x',0)
            costs = [cost, accuracy, lin_outputs]
        else:
            costs = [cost, accuracy]

    return [x, y], parameters, costs




def cue_RNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE', n_reflections=None, flag_telescope=True):
    # composes n_reflections Householder reflection matrices to make W, defaults to telescoping
    # Householder matrices, which is related to the subgroup algorithm.

    if n_reflections is None:
        # use n_hidden reflections by default (samples entire unitary group)
        n_reflections=n_hidden

    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V = initialize_matrix(n_input, 2*n_hidden, 'V', rng)
    U = initialize_matrix(2 * n_hidden, n_output, 'U', rng)
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX),
                                name='hidden_bias')

    reflection = initialize_matrix(n_reflections, 2*n_hidden, 'reflection', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')

    bucket = np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)),
                                   dtype=theano.config.floatX),
                        name='h_0')

    parameters = [V, U, hidden_bias, reflection, out_bias, h_0]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)

    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))

    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, hidden_bias, out_bias, U):

        # Compute hidden linear transform
        # def apply_reflection(ireflection, rinput, n_hidden, reflection):
        #     return times_reflection(rinput, n_hidden, reflection[ireflection,:])
        #
        # outputs_info=[h_prev]
        # sequences=[np.arange(n_reflections)]
        # non_sequences=[n_hidden,reflection]
        #
        # hidden_lin_output = theano.scan(fn=apply_reflection,
        #                                 outputs_info=outputs_info,
        #                                 sequences=sequences,
        #                                 non_sequences=non_sequences)
        # hidden_lin_output = hidden_lin_output[-1]

        step=h_prev
        #for ii in range(0,n_reflections):
        #    step=times_reflection(step, n_hidden, reflection[ii,:])
        for ii in range(n_hidden,n_hidden-n_reflections,-1):
            if flag_telescope:
                step=times_reflection_sub(step, n_hidden, ii, reflection[ii-1,:])
            else:
                step=times_reflection(step, n_hidden, reflection[ii-1,:])

        hidden_lin_output = step

        # Compute data linear transform
        if loss_function == 'CE':
            data_lin_output = V[T.cast(x_t, 'int32')]
        else:
            data_lin_output = T.dot(x_t, V)

        # Total linear output
        lin_output = hidden_lin_output + data_lin_output


        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output**2 + lin_output[:, swap_re_im]**2)
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
        h_t = lin_output * rescale

        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))

        return h_t, cost_t, acc_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [V, hidden_bias, out_bias, U]
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]

    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return [x, y], parameters, costs



 

