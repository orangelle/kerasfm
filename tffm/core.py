import tensorflow as tf
from . import utils
import math
import sys

class TFFMLayer(tf.keras.layers.Layer):
    def __init__(self, order=2, rank=2, init_std=0.01, use_diag=False, input_type='dense', reweight_reg=False):
        super(TFFMLayer, self).__init__()
        self.order = order
        self.rank = rank
        self.use_diag = use_diag
        self.init_std = init_std
        self.reweight_reg = reweight_reg
        self.input_type = input_type
        self.w = None
        self.b = None
    
    def init_learnable_params(self, n_features):
        if self.w is None:
            self.w = [None] * self.order
            for i in range(1, self.order + 1):
                r = self.rank
                if i == 1:
                    r = 1
                rnd_weights = tf.random_uniform_initializer(minval = -self.init_std, maxval = self.init_std) 
                self.w[i - 1] = self.add_weight(initializer=rnd_weights, trainable=True, name='embedding_' + str(i), shape=[n_features, r])
            self.b = tf.Variable(self.init_std, trainable=True, name='bias')

    def pow_matmul(self, inputs, order, pow):
        if pow not in self.x_pow_cache:
            x_pow = utils.pow_wrapper(inputs, pow, self.input_type)
            self.x_pow_cache[str(pow)] = x_pow
        if order not in self.matmul_cache:
            self.matmul_cache[str(order)] = {}
        if pow not in self.matmul_cache[str(order)]:
            w_pow = tf.pow(self.w[order - 1], pow)
            dot = utils.matmul_wrapper(self.x_pow_cache[str(pow)], w_pow, self.input_type)
            self.matmul_cache[str(order)][str(pow)] = dot
        return self.matmul_cache[str(order)][str(pow)]

    def call(self, inputs):
        if self.w is None:
            self.init_learnable_params(inputs.shape[1])
        
        # Add regularization
        with tf.name_scope('regularization') as scope:
            regularization = 0
            with tf.name_scope('reweights') as scope:
                if self.reweight_reg:
                    counts = utils.count_nonzero_wrapper(inputs, self.input_type)
                    sqrt_counts = tf.transpose(tf.sqrt(tf.cast(counts, tf.float32)))
                else:
                    sqrt_counts = tf.ones_like(self.w[0])
                reweights = sqrt_counts / tf.reduce_sum(sqrt_counts)
            for order in range(1, self.order + 1):
                # node_name = 'regularization_penalty_' + str(order)
                norm = tf.reduce_mean(tf.pow(self.w[order - 1]*reweights, 2))
                regularization += norm
        self.add_loss(regularization)
        

        # Outputs
        assert self.b is not None, 'n_feature is not set.'
        self.x_pow_cache = {}
        self.matmul_cache = {}
        outputs =  0.0 + self.b
        contribution = utils.matmul_wrapper(inputs, self.w[0], self.input_type)
        outputs += contribution
        for i in range(2, self.order + 1): 
            raw_dot = utils.matmul_wrapper(inputs, self.w[i - 1], self.input_type)
            dot = tf.pow(raw_dot, i)
            if self.use_diag:
                contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                contribution /= 2.0**(i-1)
            else:
                initialization_shape = tf.shape(dot)
                for in_pows, out_pows, coef in utils.powers_and_coefs(i):
                    product_of_pows = tf.ones(initialization_shape)
                    for pow_idx in range(len(in_pows)):
                        pmm = self.pow_matmul(inputs, i, in_pows[pow_idx])
                        product_of_pows *= tf.pow(pmm, out_pows[pow_idx])
                    dot -= coef * product_of_pows
                contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                contribution /= float(math.factorial(i))
            outputs += contribution
        return outputs

class TFFMCore(tf.keras.Model):
    
    def __init__(self, **layer_arguments): 
        super(TFFMCore, self).__init__()
        self.fmlayer = TFFMLayer(**layer_arguments)
    
    def init_learnable_params(self, n_features):
        self.fmlayer.init_learnable_params(n_features)
    
    def call(self, inputs):
        output = self.fmlayer(inputs)
        # print(self.fmlayer.weights)
        # self.trainable_weights_ = [*self.fmlayer.w, self.fmlayer.b]
        return output


