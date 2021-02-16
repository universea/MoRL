
class AgentBase(object):
    """`AgentBase` is the base class of the `morl.Agent` in different frameworks.

    `morl.Agent` is responsible for the general data flow outside the algorithm.
    """

    def __init__(self, algorithm):
        """

        Args:
            algorithm (`AlgorithmBase`): an instance of `AlgorithmBase`
        """
        self.alg = algorithm

    def get_weights(self, *args, **kwargs):
        """Get weights of the agent.
        
        Returns:
            (Dict): Dict of weights ({attribute name: numpy array/List/Dict})
        """
        return self.alg.get_weights(*args, **kwargs)

    def set_weights(self, weights, *args, **kwargs):
        """Set weights of the agent with given weights.

        Args:
            weights (Dict): Dict of weights
        """
        self.alg.set_weights(weights, *args, **kwargs)

    def learn(self, *args, **kwargs):
        """The training interface for Agent.
        
        This function will usually do the following things:
            1. Accept numpy data as input;
            2. Feed numpy data or onvert numpy data to tensor (optional);
            3. Call learn function in `Algorithm`.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict the action when given the observation of the enviroment.

        In general, this function is used in test process.

        This function will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data or onvert numpy data to tensor (optional);
           3. Call predict function in `Algorithm`.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Sample the action when given the observation of the enviroment.
            
        In general, this function is used in train process.

        This function will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data or onvert numpy data to tensor (optional);
           3. Call predict or sample function in `Algorithm`;
           4. Add sampling operation in numpy level. (unnecessary if sampling operation have done in `Algorithm`).
        """
        raise NotImplementedError
