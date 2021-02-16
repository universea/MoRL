
class ModelBase(object):
    """`ModelBase` is the base class of the `morl.Model` in different frameworks.

    This base class mainly do the following things:
        1. Implements APIs to manage model_id of the `morl.Model`; 
        2. Defines common APIs that `morl.Model` should implement in different frameworks.
    """

    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        """Define forward network of the model.
        """
        raise NotImplementedError

    def get_weights(self, *args, **kwargs):
        """Get weights of the model.
        """
        raise NotImplementedError

    def set_weights(self, weights, *args, **kwargs):
        """Set weights of the model with given weights.
        """
        raise NotImplementedError

    def sync_weights_to(self, other_model):
        """Synchronize weights of the model to another model.
        """
        raise NotImplementedError

    def parameters(self):
        """Get the parameters of the model.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Call forward function.
        """
        return self.forward(*args, **kwargs)
