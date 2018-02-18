from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer


class Modified_SGD(Optimizer):
    """ Modified Stochastic gradient descent optimizer.

    Almost all this class is Keras SGD class code. I just reorganized it
    in this class to allow layer-wise momentum and learning-rate

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    Includes the possibility to add multipliers to different
    learning rates in each layer.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        lr_multipliers: dictionary with learning rate for a specific layer
        for example:
            # Setting the Learning rate multipliers
            LR_mult_dict = {}
            LR_mult_dict['c1']=1
            LR_mult_dict['c2']=1
            LR_mult_dict['d1']=2
            LR_mult_dict['d2']=2
        momentum_multipliers: dictionary with momentum for a specific layer 
        (similar to the lr_multipliers)
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, lr_multipliers=None, momentum_multipliers=None, **kwargs):
        super(Modified_SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.lr_multipliers = lr_multipliers
        self.momentum_multipliers = momentum_multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        
        
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):

            if self.lr_multipliers != None:
                if p.name in self.lr_multipliers:
                    new_lr = lr * self.lr_multipliers[p.name]
                else:
                    new_lr = lr
            else:
                new_lr = lr

            if self.momentum_multipliers != None:
                if p.name in self.momentum_multipliers:
                    new_momentum = self.momentum * \
                        self.momentum_multipliers[p.name]
                else:
                    new_momentum = self.momentum
            else:
                new_momentum = self.momentum

            v = new_momentum * m - new_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + new_momentum * v - new_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'lr_multipliers': float(K.get_value(self.lr_multipliers)),
                  'momentum_multipliers': float(K.get_value(self.momentum_multipliers))}
        base_config = super(Modified_SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
