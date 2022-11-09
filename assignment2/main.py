import itertools
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

class RNNStateSet:
    def __init__(self, dimension: int, gibbs_total: int, weights: list, const: float, gain: float):
        """
        :param dimension: dimension of the state
        :param gibbs_total:
        :param weights: list of weights[dimension + 1][dimension + 1]
        :param const:
        """
        self.state_total = 2 ** dimension
        self.state_sets = self.__init_state_sets(dimension, gibbs_total)
        self.gibbs_total = gibbs_total
        self.weights = weights
        self.const = const
        self.gain = gain
        self.__set_all_energy(weights, const)
        self.A = self.__calc_gibbs_const()
        self.__set_all_theoretical_gibbs_num()
        self.temporal_transition_record = [0]
        self.update_count = 0
        self.binary_update_order = None

    def get_index(self, binary_set):
        for idx, st in enumerate(self.state_sets):
            if binary_set == st.binary_set:
                return idx

    def update_gibbs_system(self, update_num: int, model: str="probabilistic", binary_update_order: list="None"):
        """
        if model == "probabilistic":
            binary_update_order = None
        elif model == "deterministic":
            binary_update_order = list[dimension]
            # [0, 1, 2, ... n-1]
        """
        for _ in range(update_num):
            self.__update_gibbs_system_(model, binary_update_order)
            self.update_count += 1
        self.__set_transition_count()

    def __update_gibbs_system_(self, model: str, binary_update_order: list=None):
        if model == "probabilistic":
            model = ProbabilisticModel(self.gain, self.weights)
        elif model == "deterministic":
            model = DeterministicModel(self.gain, self.weights, self.update_count, binary_update_order)
        else:
            raise ValueError("model must be probabilistic or deterministic")

        state_transition_record = [0] * self.state_total
        for state_index in range(self.state_total):
            for loop_count in range(self.state_sets[state_index].gibbs_num):
                binary_set_after_transition = model.pred(self.state_sets[state_index].binary_set)
                state_transition_record[self.get_index(binary_set_after_transition)] += 1
                if (loop_count == 0) & (self.temporal_transition_record[self.update_count] == state_index):
                    self.temporal_transition_record.append(self.get_index(binary_set_after_transition))

        for index, gibbs_num in enumerate(state_transition_record):
            self.__set_gibbs_num(index, gibbs_num)

    def __set_transition_count(self):
        transition_count_set = collections.Counter(self.temporal_transition_record)
        for key, value in transition_count_set.items():
            self.state_sets[key].transition_count = value

    def __init_state_sets(self, dimension, gibbs_total):
        state_sets = list()
        binary_sets = self.__create_state_set(dimension)
        gibbs_num_list = self.__init_gibbs(gibbs_total)
        energy_list = self.__init_energy()

        for binary_set, gibbs_num, energy in zip(binary_sets, gibbs_num_list, energy_list):
            rnn_state = _RNNState(binary_set, gibbs_num, energy)
            state_sets.append(rnn_state)
        return state_sets

    def __set_gibbs_num(self, index, gibbs_num):
            self.state_sets[index].gibbs_num = gibbs_num

    def __calc_gibbs_const(self):
        return self.gibbs_total / sum([np.exp((-1) * self.gain * state_set.energy) for state_set in self.state_sets])

    def __create_state_set(self, dimension):
        return [set for set in itertools.product('01', repeat=dimension)]

    def __init_gibbs(self, gibbs_total):
        gibbs_num = [0] * self.state_total
        gibbs_num[0] = gibbs_total
        return gibbs_num

    def __init_energy(self):
        return [0] * self.state_total

    def __set_all_energy(self, weights: list, const: float):
        for state_index, state_set in enumerate(self.state_sets):
            energy = self.__calc_energy(state_set.binary_set, weights, const)
            self.__set_energy(state_index, energy)

    def __calc_energy(self, binary_set: tuple, weights: list, const: float):
        neurons = np.array([1] + list(map(int, binary_set)))
        weights = np.array(weights)
        return (-1) * (1 / 2) * np.sum(weights * (neurons.reshape(1, -1).T @ neurons.reshape(1, -1))) + const

    def __set_energy(self, state_index, energy):
        self.state_sets[state_index].energy = energy

    def __set_all_theoretical_gibbs_num(self):
        for state_index, state_set in enumerate(self.state_sets):
            N = self.__calc_theoretical_gibbs_num(state_set.energy)
            self.__set_theoretical_gibbs_num(state_index, N)

    def __calc_theoretical_gibbs_num(self, energy):
        return self.A * np.exp((-1) * self.gain * energy)

    def __set_theoretical_gibbs_num(self, state_index, N):
        self.state_sets[state_index].N = N

class _RNNState:
    def __init__(self, binary_set, gibbs_num, energy):
        self.binary_set = binary_set
        self.gibbs_num = gibbs_num
        self.N = 0
        self.energy = energy
        self.transition_count = 0

class BinaryModel:
    def __init__(self, gain: float, weights: list):
        self.gain = gain
        self.weights = weights

    def pred(self, input):
        pass

    def _activation(self, s_hat):
        pass

    def __sigmoid(self, alpha, s_hat):
        pass

    def __decision_binary(self, p):
        pass

class DeterministicModel(BinaryModel):
    def __init__(self, gain: float, weights: list, update_count: int, binary_update_order: list):
        super().__init__(gain, weights)
        self.update_count = update_count
        self.binary_update_order = binary_update_order

    def pred(self, input: tuple):
        input_list_casting = list(map(int, input))
        weights = np.array(self.weights)
        input_with_bias = np.array([1] + input_list_casting)

        # choose the update neuron
        update_index = self.__choice_neuron(input)
        s_hat = np.sum(weights[(update_index + 1), :] * (input_with_bias))
        input_list_casting[update_index] = self._activation(s_hat)
        return tuple(map(str, input_list_casting))

    def __choice_neuron(self, input: tuple):
        dimension = len(input)
        update_index = self.binary_update_order[self.update_count % dimension]
        return update_index

    def _activation(self, s_hat):
        p = self.__sigmoid(self.gain, s_hat)
        y = self.__decision_binary(p)
        return y

    def __sigmoid(self, alpha, s_hat):
        p = 1 / (1 + np.exp((-1) * alpha * s_hat))
        return p

    def __decision_binary(self, p):
        if p >= (1 / 2):
            return 1
        else:
            return 0

class ProbabilisticModel(BinaryModel):
    def __init__(self, gain: float, weights: list):
        self.gain = gain
        self.weights = weights

    def pred(self, input):
        input_list_casting = list(map(int, input))
        weights = np.array(self.weights)
        input_with_bias = np.array([1] + input_list_casting)

        # choose the update neuron
        update_index = random.choices(range(len(input)), k=1, weights=[1, 1, 1])[0]
        s_hat = np.sum(weights[(update_index + 1), :] * (input_with_bias))
        input_list_casting[update_index] = self._activation(s_hat)
        return tuple(map(str, input_list_casting))

    def _activation(self, s_hat):
        p = self.__sigmoid(self.gain, s_hat)
        y = self.__decision_binary(p)
        return y

    def __sigmoid(self, alpha, s_hat):
        p = 1 / (1 + np.exp((-1) * alpha * s_hat))
        return p

    def __decision_binary(self, p):
        prob_weights = [1-p, p]
        return random.choices([0, 1], weights=prob_weights, k=1)[0]

def main():
    dimension = 3
    gibbs_total = 1
    weights = [
        [0, 6, -16, 5],
        [6, 0, 10, -8],
        [-16, 10, 0, 8],
        [5, -8, 8, 0]
    ]
    const = 6
    gain = 0.1
    rnn = RNNStateSet(dimension, gibbs_total, weights, const, gain)
    # rnn.update_gibbs_system(update_num=100, model="probabilistic")

    deterministic_order = [list(tup) for tup in itertools.permutations(range(dimension))]
    # print(deterministic_order)
    for j, dorder in enumerate(deterministic_order):
        # print(dorder)
        data_matrix = [['update', '$x1, x2, x3$', 'energy']]
        for i in range(5):
            rnn = RNNStateSet(dimension, gibbs_total, weights, const, gain)
            rnn.update_gibbs_system(update_num=5, model="deterministic", binary_update_order=dorder)
            # print(f"{rnn.state_sets[i].binary_set}: ", rnn.state_sets[i].energy)
            data_matrix.append([i, list(map(int, rnn.state_sets[rnn.temporal_transition_record[i]].binary_set)), rnn.state_sets[rnn.temporal_transition_record[i]].energy])
        # print(data_matrix)
        fig = ff.create_table(data_matrix)
        # print(index)
        # fig.write_image(f'./img/table_0_{j}.png')
        import sys
        sys.exit()
        # import time
        # time.sleep(2)
        # fig.show()

    import sys
    sys.exit()

    print(f"the number of updating system: {rnn.update_count}")
    for index in range(rnn.state_total):
        print(
            rnn.state_sets[index].binary_set,
            rnn.state_sets[index].energy,
            rnn.state_sets[index].gibbs_num,
            rnn.state_sets[index].N,
            rnn.state_sets[index].transition_count
            )

    x_theory = np.linspace(-1, 25, 1000)
    y_theory = rnn.A * np.exp((-1) * rnn.gain * x_theory)

    x_gibbs = np.array([state_set.energy for state_set in rnn.state_sets])
    y_gibbs = np.array([state_set.gibbs_num for state_set in rnn.state_sets])

    x_transiton = np.array([state_set.energy for state_set in rnn.state_sets])
    y_transiton = np.array([(rnn.gibbs_total / rnn.update_count) * float(state_set.transition_count) for state_set in rnn.state_sets])

    label = np.array(["(" + ",".join(list(state_set.binary_set)) + ")" for state_set in rnn.state_sets])


    fig, ax = plt.subplots(figsize=(10, 4))
    for (i, j, k) in zip(x_gibbs, y_gibbs, label):
        plt.scatter(i, j, s=30, c="#8d70ff", alpha=0.5, linewidths=1,
            edgecolors="#44008c", zorder=1, label="the number of gibbs copy")
        plt.annotate(k, xy=(i, j), color="#44008c", zorder=1)
    for (i, j, k) in zip(x_transiton, y_transiton, label):
        plt.scatter(i, j, s=30, c="#a13248", alpha=0.5, linewidths=1,
            edgecolors="#6b202f", zorder=1, label="transition count")
        plt.annotate(k, xy=(i, j), color="#6b202f", zorder=1)
    plt.plot(x_theory, y_theory, c="#380227", lw=1, zorder=2, label="theoritical value")
    plt.title("Boltzman Distribution")
    plt.xlabel("E")
    plt.ylabel("N")
    plt.xlim([-1, 25])
    plt.xticks(rotation=90)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0], handles[1], handles[9]], [labels[0], labels[1], labels[9]])
    plt.show()

if __name__ == "__main__":
    main()