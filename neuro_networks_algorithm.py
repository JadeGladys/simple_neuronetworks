import random
import math

# SETTINGS - numbers you choose yourself
DATA_FILE = "iris.data"
TRAIN_SHARE = 0.6       # 60% of flowers for training, 40% for testing
SEED = 42
HIDDEN1_NEURONS = 10    # neurons in hidden layer 1
HIDDEN2_NEURONS = 10    # neurons in hidden layer 2
START_RANGE = 0.5       # starting weights and biases are random between -0.5 and 0.5


# =====================================================================
# PART 1 - Load the iris data
# =====================================================================
def load_flowers(file_name):
    """Return a list of flowers. Each flower is (measurements, species)."""
    flowers = []
    with open(file_name, newline="", encoding="utf-8") as file:

        # record measurement and species per row
        for line in file:
            line = line.strip()
            if line == "":
                continue

            parts = line.split(",")

            measurements = []

            for num in parts[0:4]:
                measurements.append(float(num))

            species = parts[4]
            flowers.append((measurements, species))
    return flowers


# =====================================================================
# PART 2 - prepare the data (shuffle, split into training/testing)
# =====================================================================
SPECIES_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def species_to_true_answer(species):
    """Turn a species name into its true answer list, e.g. "versicolor" -> [0, 1, 0]"""
    true_answer = [0, 0, 0]
    position = SPECIES_NAMES.index(species)
    true_answer[position] = 1
    return true_answer


def prepare_flowers(flowers, train_share, seed):
    """Return (training, testing). Each is (measurements, true answer)."""
    rdm = random.Random(seed)

    prepared = []
    for measurements, species in flowers:
        prepared.append((measurements, species_to_true_answer(species)))

    # shuffle the list to make sure the flower list is not in order
    rdm.shuffle(prepared)

    train_count = int(len(prepared) * train_share)
    training = prepared[:train_count]
    testing = prepared[train_count:]

    return training, testing


# =====================================================================
# PART 3 - create the network (random starting weights and biases)
# =====================================================================
INPUT_SIZE = 4     # 4 measurements per flower (fixed by the data)
OUTPUT_SIZE = 3    # output layer has only 3 neurons for the 3 species


def create_layer(number_of_inputs, number_of_neurons, start_range, rdm):
    """Return one layer with random weights and biases."""
    weights = []
    for neuron in range(number_of_neurons):
        this_neuron_weights = []
        for _ in range(number_of_inputs):
            this_neuron_weights.append(rdm.uniform(-start_range, start_range))
        weights.append(this_neuron_weights)

    biases = []
    for neuron in range(number_of_neurons):
        biases.append(rdm.uniform(-start_range, start_range))

    return {"weights": weights, "biases": biases}


def create_network(hidden1_neurons, hidden2_neurons, start_range, seed):
    """Return the 3 layers: 4 inputs -> hidden 1 inputs -> hidden 2 inputs -> 3 outputs."""
    rdm = random.Random(seed)

    network = {
        "hidden_layer1": create_layer(INPUT_SIZE, hidden1_neurons, start_range, rdm),
        "hidden_layer2": create_layer(hidden1_neurons, hidden2_neurons, start_range, rdm),
        "output_layer": create_layer(hidden2_neurons, OUTPUT_SIZE, start_range, rdm)
    }

    return network


# =====================================================================
# PART 4 - forward pass
# measurements -> hidden 1 -> hidden 2 -> output scores -> probabilities
# =====================================================================
def calculate_totals(layer, inputs):
    """For every neuron: total = (each input x its weight, added up) + bias."""
    totals = []
    for neuron in range(len(layer["weights"])):
        total = layer["biases"][neuron]
        for input_index in range(len(inputs)):
            total += inputs[input_index] * \
                layer["weights"][neuron][input_index]
        totals.append(total)

    return totals


def relu(totals):
    """Negative totals become 0, positive totals stay the same."""
    outputs = []

    for total in totals:
        if total > 0:
            outputs.append(total)
        else:
            outputs.append(0.0)

    return outputs


def softmax(scores):
    """Turn the scores into probabilities that add up to 1."""

    exponents = []
    for score in scores:
        exponents.append(math.exp(score))   # e^score

    exponents_sum = sum(exponents)

    probabilities = []
    for exponent in exponents:
        probabilities.append(exponent / exponents_sum)

    return probabilities


def forward_pass(network, measurements):
    """Run one flower through the network. Save everything the backward pass needs."""
    hidden1_totals = calculate_totals(network["hidden_layer1"], measurements)
    hidden1_outputs = relu(hidden1_totals)

    hidden2_totals = calculate_totals(
        network["hidden_layer2"], hidden1_outputs)
    hidden2_outputs = relu(hidden2_totals)

    # no ReLU on the output layer
    scores = calculate_totals(network["output_layer"], hidden2_outputs)
    probabilities = softmax(scores)

    return {
        "measurements": measurements,
        "hidden1_totals": hidden1_totals,
        "hidden1_outputs": hidden1_outputs,
        "hidden2_totals": hidden2_totals,
        "hidden2_outputs": hidden2_outputs,
        "probabilities": probabilities,
    }
