
import os
import paddle
from morl.core.agent_base import AgentBase
from morl.core.paddle.algorithm import Algorithm
from morl.utils import machine_info, get_gpu_count

__all__ = ['Agent']


class Agent(AgentBase):
    """
    | `alias`: ``morl.Agent``
    | `alias`: ``morl.core.paddle.agent.Agent``

    | Agent is one of the three basic classes of morl.

    | It is responsible for interacting with the environment and collecting 
    data for training the policy.
    | To implement a customized ``Agent``, users can:

      .. code-block:: python

        import morl

        class MyAgent(morl.Agent):
            def __init__(self, algorithm, act_dim):
                super(MyAgent, self).__init__(algorithm)
                self.act_dim = act_dim

    Attributes:
        alg (morl.algorithm): algorithm of this agent.
        place: can automatically specify device when creating a tensor.

    Public Functions:
        - ``sample``: return a noisy action to perform exploration according to the policy.
        - ``predict``: return an action given current observation.
        - ``learn``: update the parameters of self.alg using the `learn_program` defined in `build_program()`.
        - ``save``: save parameters of the ``agent`` to a given path.
        - ``restore``: restore previous saved parameters from a given path.

    Todo:
        - allow users to get parameters of a specified model by specifying the model's name in ``get_weights()``.

    """

    def __init__(self, algorithm):
        """

        Args:
            algorithm (morl.Algorithm): an instance of `morl.Algorithm`. This algorithm is then passed to `self.alg`.
        """

        assert isinstance(algorithm, Algorithm)
        super(Agent, self).__init__(algorithm)

        gpu_count = get_gpu_count()
        if gpu_count > 0:
            self.place = paddle.CUDAPlace(0)
        else:
            self.place = paddle.CPUPlace()

    def learn(self, *args, **kwargs):
        """The training interface for ``Agent``.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict an action when given the observation of the environment.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Return an action with noise when given the observation of the environment.

        In general, this function is used in train process as noise is added to the action to preform exploration.

        """
        raise NotImplementedError

    def save(self, save_path, model=None):
        """Save parameters.

        Args:
            save_path(str): where to save the parameters.
            model(morl.Model): model that describes the neural network structure. If None, will use self.alg.model.

        Raises:
            ValueError: if program is None and self.learn_program does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model_dir')

        """
        if model is None:
            model = self.alg.model
        paddle.save(model.state_dict(), save_path)

    def restore(self, save_path, model=None):
        """Restore previously saved parameters.
        This method requires a program that describes the network structure.
        The save_path argument is typically a value previously passed to ``save_params()``.

        Args:
            save_path(str): path where parameters were previously saved.
            model(morl.Model): model that describes the neural network structure. If None, will use self.alg.model.

        Raises:
            ValueError: if program is None and self.learn_program does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model_dir')
            agent.restore('./model_dir')

        """
        if model is None:
            model = self.alg.model
        param_dict = paddle.load(save_path)
        model.set_state_dict(param_dict)
