
from morl.core.algorithm_base import AlgorithmBase
from morl.core.paddle.model import Model

__all__ = ['Algorithm']


class Algorithm(AlgorithmBase):
    """
    | `alias`: ``morl.Algorithm``
    | `alias`: ``morl.core.fluid.algorithm.Algorithm``

    | ``Algorithm`` defines the way how to update the parameters of 
    the ``Model``. This is where we define loss functions and the 
    optimizer of the neural network. An ``Algorithm`` has at least 
    a model.

    | morl has implemented various algorithms(DQN/DDPG/PPO/A3C/IMPALA) that 
    can be reused quickly, which can be accessed with ``morl.algorithms``.

    Example:

    .. code-block:: python

        import morl

        model = Model()
        dqn = morl.algorithms.DQN(model, lr=1e-3)

    Attributes:
        model(``morl.Model``): a neural network that represents a policy 
        or a Q-value function.

    Pulic Functions:
        - ``get_weights``: return a Python dictionary containing parameters 
        of the current model.
        - ``set_weights``: copy parameters from ``get_weights()`` to the model.
        - ``sample``: return a noisy action to perform exploration according 
        to the policy.
        - ``predict``: return an action given current observation.
        - ``learn``: define the loss function and create an optimizer to 
        minized the loss.
    """

    def __init__(self, model=None):
        """
        Args:
            model(``morl.Model``): a neural network that represents a policy or a Q-value function.
        """
        assert isinstance(model, Model)
        self.model = model

    def learn(self, *args, **kwargs):
        """ Define the loss function and create an optimizer to minize the loss.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """ Refine the predicting process, e.g,. use the policy model to predict actions.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """ Define the sampling process. This function returns an action with noise to perform exploration.
        """
        raise NotImplementedError

    def get_weights(self):
        """ Get weights of self.model.

        Returns:
            weights (dict): a Python dict containing the parameters of self.model.
        """
        return self.model.get_weights()

    def set_weights(self, params):
        """ Set weights from ``get_weights`` to the model.

        Args:
            weights (dict): a Python dict containing the parameters of self.model.
        """
        self.model.set_weights(params)
