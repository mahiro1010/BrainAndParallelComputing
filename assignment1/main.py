import numpy as np
import random

class ProbabilisticBinaryNeuron:

    def __init__(self, gain):
        self.gain = gain

    def pred(self, weight, input):
        """
        Parameters
        ----------
        input : list[int]
            [x0(bias), x1, x2,... ,xN]
            xk in [0, 1] 
        Returns
        -------
        int
            return y
            y in [0, 1]
        """
        s_hat = np.inner(weight, input)
        return self.__activation(s_hat)

    def __activation(self, s_hat):
        p = self.__sigmoid(self.gain, s_hat)
        y = self.__decision_binary(p)
        return p, y

    def __sigmoid(self, alpha, s_hat):
        p = 1 / (1 + np.exp((-1) * alpha * s_hat))
        return p

    def __decision_binary(self, p):
        prob_weights = [1-p, p]
        return random.choices([0, 1], weights=prob_weights, k=1)[0]

def main():
    INPUT_NUM = 5
    GAINS = [0.1, 1, 2, 5]
    TRIALS = [100, 1000, 10000]
    input = np.array(random.choices([0, 1], weights=[0.5, 0.5], k=INPUT_NUM))
    weight = np.random.rand(INPUT_NUM) * 2 - 1

    for gain in GAINS:
        neuron = ProbabilisticBinaryNeuron(gain)
        for trial in TRIALS:
            count = 0
            for _ in range(trial):
                p, y = neuron.pred(weight, input)
                count += y
            print(f"gain: {gain}, trial: {trial}, probability: {p}, ratio of the value 1 : {count/trial}")

if __name__ == "__main__":
    main()