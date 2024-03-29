import tensorflow as tf
from .core import TFFMCore
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from abc import ABCMeta, abstractmethod
import six
from tqdm import tqdm
import numpy as np
import os


def batcher(X_, y_=None, w_=None, batch_size=-1):
    """Split data to mini-batches.

    Parameters
    ----------
    X_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_ : np.array or None, shape (n_samples,)
        Target vector relative to X.

    w_ : np.array or None, shape (n_samples,)
        Vector of sample weights.

    batch_size : int
        Size of batches.
        Use -1 for full-size batches

    Yields
    -------
    ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Same type as input

    ret_y : np.array or None, shape (batch_size,)

    ret_w : np.array or None, shape (batch_size,)
    """
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        ret_w = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
        if w_ is not None:
            ret_w = w_[i:i + batch_size]
        yield (ret_x, ret_y, ret_w)


def batch_feed(X, input_type):
    """Prepare inputs for session.run() from mini-batch.
    Convert sparse format into tuple (indices, values, shape) for tf.SparseTensor
    ----------
    Parameters
    ----------
    X : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Training vector, where batch_size in the number of samples and
        n_features is the number of features.

    -------
    Returns
    -------
    inputs : {tf.Tenfor, tf.SparseTensor}, shape (batch_size, n_features)
        Dict with formatted placeholders
    """
    if input_type == 'dense':
        inputs = tf.convert_to_tensor(X.astype(np.float32), dtype = tf.float32)
        return inputs
    else:
        # sparse case
        X_sparse = X.tocoo()
        raw_indices = np.hstack(
            (X_sparse.row[:, np.newaxis], X_sparse.col[:, np.newaxis])
        ).astype(np.int64)
        raw_values = X_sparse.data.astype(np.float32)
        raw_shape = np.array(X_sparse.shape).astype(np.int64)
        raw_indices = tf.convert_to_tensor(raw_indices, dtype = tf.int64)
        raw_values = tf.convert_to_tensor(raw_values, dtype = tf.float32)
        raw_shape = tf.convert_to_tensor(raw_shape, dtype = tf.int64)
        inputs = tf.SparseTensor(raw_indices, raw_values, raw_shape)
        return inputs



class TFFMBaseModel(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for FM.
    This class implements L2-regularized arbitrary order FM model.

    It supports arbitrary order of interactions and has linear complexity in the
    number of features (a generalization of the approach described in Lemma 3.1
    in the referenced paper, details will be added soon).

    It can handle both dense and sparse input. Only numpy.array and CSR matrix are
    allowed as inputs; any other input format should be explicitly converted.

    Support logging/visualization with TensorBoard.

    ----------
    Parameters (for initialization)
    ----------
    batch_size : int, default: -1
        Number of samples in mini-batches. Shuffled every epoch.
        Use -1 for full gradient (whole training set in each batch).

    n_epoch : int, default: 100
        Default number of epoches.
        It can be overrived by explicitly provided value in fit() method.

    log_dir : str or None, default: None
        Path for storing model stats during training. Used only if is not None.
        WARNING: If such directory already exists, it will be removed!
        You can use TensorBoard to visualize the stats:
        `tensorboard --logdir={log_dir}`

    session_config : tf.ConfigProto or None, default: None
        Additional setting passed to tf.Session object.
        Useful for CPU/GPU switching, setting number of threads and so on,
        `tf.ConfigProto(device_count = {'GPU': 0})` will disable GPU (if enabled)

    verbose : int, default: 0
        Level of verbosity.
        Set 1 for tensorboard info only and 2 for additional stats every epoch.
    
    reg: flaot, default: 0
        Strength of L2 regularization

    optimizer: tf.keras.optimizers.Optimizer, default: Adam(learning_rate=0.01)
        Optimizer used for training

    loss_function : function: (tf.Op, tf.Op) -> tf.Op, default: None
        Loss function.
        Take 2 tf.Ops: outputs and targets and should return tf.Op of loss
        See examples: .utils.loss_mse, .utils.loss_logistic

    kwargs : dict, default: {}
        Arguments for TFFMCore constructor.
        See TFFMCore's doc for details.

    seed : int or None, default: None
        Random seed used at core model creating time

    ----------
    Attributes
    ----------
    core : TFFMCore or None
        The core model 
        Will be initialized during first call .fit()

    steps : int
        Counter of passed lerning epochs, used as step number for writing stats

    intercept : float, shape: [1]
        Intercept (bias) term.

    weights : array of np.array, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    """


    def init_basemodel(self, n_epochs=100, batch_size=-1, log_dir=None, verbose=0, 
                       seed=None, pos_class_weight=None, input_type="dense", reg=0, 
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                       loss_function=None, **core_arguments):
        core_arguments['input_type'] = input_type
        self.core = TFFMCore(**core_arguments)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.verbose = verbose
        self.seed = seed
        self.pos_class_weight = pos_class_weight
        self.optimizer = optimizer
        self.input_type = input_type
        self.loss_function = loss_function
        self.reg = reg
        self.steps = 0
        self.target = 0
        self.reduced_loss = 0
        self.regularization = 0


    def initialize_summary(self):
        """Initialize summary logger (if needed).
        """
        if self.need_logs:
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            if self.verbose > 0:
                full_log_path = os.path.abspath(self.log_dir)
                print('Initialize logs, use: \ntensorboard --logdir={}'.format(full_log_path))

    def execute_summary(self, step):
        """Write summary data with built writer
        """
        with self.summary_writer.as_default():
            tf.summary.scalar('bias', self.core.fmlayer.b, step=step)
            tf.summary.scalar('regularization_penalty', self.regularization, step=step)
            tf.summary.scalar('loss', self.reduced_loss, step=step)
            tf.summary.scalar('target', self.target, step=step)

    @tf.function
    def train(self, inputs, labels, sample_weights, steps):
        """Graph function for training the model
        A graph is built and the training speed is boosted 
        """
        with tf.GradientTape() as tape:
            pred = self.core(inputs)
            loss = self.loss_function(pred, labels) * sample_weights
            self.reduced_loss = tf.reduce_mean(loss)
            self.regularization = self.reg * sum(self.core.losses)
            self.target = self.reduced_loss + self.regularization
        grads = tape.gradient(self.target, self.core.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_weights))
        if self.need_logs:
            self.execute_summary(steps)
        return self.target
    
    def train_eager(self, inputs, labels, sample_weights):
        """Eager function for training the model
        The eager training function may be much slower than the graph version,
        but it's easier to get the values of the variables in the funciton.
        """
        with tf.GradientTape() as tape:
            pred = self.core(inputs)
            loss = self.loss_function(pred, labels) * sample_weights
            self.reduced_loss = tf.reduce_mean(loss)
            self.regularization = self.reg * sum(self.core.losses)
            self.target = self.reduced_loss + self.regularization
        grads = tape.gradient(self.target, self.core.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_weights))
        # write stats 
        if self.need_logs:
            self.execute_summary(self.steps)
        return self.target

    def _fit(self, X_, y_, w_, n_epochs=None, show_progress=False):

        # Initialize the learnable weights
        self.core.init_learnable_params(X_.shape[1])

        if self.need_logs:
            self.initialize_summary()

        if n_epochs is None:
            n_epochs = self.n_epochs

        # For reproducible results
        if self.seed:
            np.random.seed(self.seed)
        
        # Training cycle
        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):
            # generate permutation
            perm = np.random.permutation(X_.shape[0])
            epoch_loss = []
            # iterate over batches
            for bX, bY, bW in batcher(X_[perm], y_=y_[perm], w_=w_[perm], batch_size=self.batch_size):
                inputs = batch_feed(bX, self.input_type)
                labels = tf.convert_to_tensor(bY.astype(np.float32), dtype = tf.float32)
                sample_weights = tf.convert_to_tensor(bW.astype(np.float32), dtype = tf.float32)
                epoch_loss.append(self.train(inputs, labels, sample_weights, tf.convert_to_tensor(self.steps,tf.int64)))
                self.summary_writer.flush()
                self.steps += 1
            if self.verbose > 1:
                    print('[epoch {}]: mean target value: {}'.format(epoch, np.mean(epoch_loss)))

    def decision_function(self, X, pred_batch_size=None, graph_mode=False):
        output = []
        if pred_batch_size is None:
            pred_batch_size = self.batch_size

        for bX, _, _ in batcher(X, y_=None, w_=None, batch_size=pred_batch_size):
            inputs = batch_feed(bX, self.input_type)
            output.append(self.core(inputs))
            
        distances = np.concatenate(output).reshape(-1)
        # WARNING: be careful with this reshape in case of multiclass
        return distances

    @abstractmethod
    def predict(self, X, pred_batch_size=None):
        """Predict target values for X."""

    @property
    def intercept(self):
        """Export bias term from tf.Variable to float."""
        return self.core.fmlayer.b.numpy()
    @property
    def weights(self):
        """Export underlying weights from tf.Variables to np.arrays."""
        return [x.numpy() for x in self.core.w]

    def save_state(self, path):
        self.core.save_weights(path)

    def load_state(self, path):
        self.core.load_weights(path)

