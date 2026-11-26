import random
import math


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


# ==========================================================================
# PART 4 - forward pass
# measurements -> hidden 1 -> hidden 2 -> output scores -> probabilities
# ==========================================================================
def calculate_totals(layer, inputs):
    """For every neuron: total = (each input x its weight, added up) + bias."""
    totals = []
    for neuron in range(len(layer["weights"])):
        total = layer["biases"][neuron]
        for input_index, input_value in enumerate(inputs):
            total += input_value * layer["weights"][neuron][input_index]
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
    biggest_score = max(scores)

    exponents = []
    for score in scores:
        # subtract the biggest score: same probabilities, but e^ never gets too big
        exponents.append(math.exp(score - biggest_score))   # e^()

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


# ========================================================================
# PART 5 - loss = -ln(probability the network gave the correct species)
# ========================================================================
def calculate_loss(probabilities, true_answer):
    """How wrong the guess was: 0 = perfect, bigger = worse."""
    correct_position = true_answer.index(1)

    # max(..., 1e-12): if the probability is exactly 0, use 0.000000000001 instead
    # math.log = ln
    return -math.log(max(probabilities[correct_position], 1e-12))


# ========================================================================
# PART 6 - backward pass
# ========================================================================
def calculate_layer_derivatives(blames, inputs):
    """Return the derivatives of one layer, in the same shape as the layer itself."""
    # calculate the derivative of each neuron N0 of blames = N0 of neurons
    weight_derivatives = []
    for _, blame in enumerate(blames):
        this_neuron_derivatives = []
        for _, input_value in enumerate(inputs):
            this_neuron_derivatives.append(blame * input_value)
        weight_derivatives.append(this_neuron_derivatives)

    bias_derivatives = []
    for _, blame in enumerate(blames):
        # bias link is 1, so it equals the blame
        bias_derivatives.append(blame)

    return {"weight_derivatives": weight_derivatives,
            "bias_derivatives": bias_derivatives}


def calculate_hidden_blames(next_layer, next_layer_blames, this_layer_totals):
    """A hidden layer calculates its own blames. using the next layer's weight and blames
    next_layer        = the layer after this one (we need its weights)
    next_layer_blames = the blames sent back from the layer after
    this_layer_totals = this layer's totals, before ReLU (for the ReLU link)
    """
    this_layer_blames = []
    for neuron, this_layer_total in enumerate(this_layer_totals):
        blame_from_next_layer = 0.0

        # we find the derivative of how this layer's output affects next layers' blame
        # the formula is next layer's weight times the blame for each neuron
        for next_neuron, next_layer_blame in enumerate(next_layer_blames):
            blame_from_next_layer += next_layer_blame * \
                next_layer["weights"][next_neuron][neuron]

        # ReLU link: 1 if this neuron's total was positive, 0 if not
        if this_layer_total > 0:
            relu_link = 1
        else:
            relu_link = 0

        this_layer_blames.append(blame_from_next_layer * relu_link)
    return this_layer_blames


def backward_pass(network, forward_pass_results, true_answer):
    """For one flower: the derivative of every weight and bias in the network."""
    # ---- output layer ----
    # output layer blame = probability - true answer
    output_layer_blame = []
    for position, true_value in enumerate(true_answer):
        output_layer_blame.append(
            forward_pass_results["probabilities"][position] - true_value)
    # output layer derivative = hidden layer output * blame
    output_derivatives = calculate_layer_derivatives(
        output_layer_blame, forward_pass_results["hidden2_outputs"])

    # ---- hidden layer 2 ----
    # blame: from the output layer's blames and weights
    hidden2_blames = calculate_hidden_blames(
        network["output_layer"], output_layer_blame, forward_pass_results["hidden2_totals"])
    # derivatives: inputs are hidden layer 1's outputs
    hidden2_derivatives = calculate_layer_derivatives(
        hidden2_blames, forward_pass_results["hidden1_outputs"])

    # ---- hidden layer 1 ----
    # 1. blame: from hidden layer 2's blames and weights
    hidden1_blames = calculate_hidden_blames(
        network["hidden_layer2"], hidden2_blames, forward_pass_results["hidden1_totals"])
    # 2. derivatives: inputs are the flower's measurements
    hidden1_derivatives = calculate_layer_derivatives(
        hidden1_blames, forward_pass_results["measurements"])

    return {
        "hidden_layer1": hidden1_derivatives,
        "hidden_layer2": hidden2_derivatives,
        "output_layer": output_derivatives,
    }


# ========================================================================
# PART 7 - update: new = old - learning rate x derivative
# ========================================================================
def update_network(network, derivatives, learning_rate):
    """Change every weight and bias a little, in the direction that lowers the loss."""
    for layer_name in ["hidden_layer1", "hidden_layer2", "output_layer"]:
        layer = network[layer_name]
        layer_derivatives = derivatives[layer_name]

        # weights: same position in the layer and in its derivatives
        for neuron in range(len(layer["weights"])):
            for input_index in range(len(layer["weights"][neuron])):
                layer["weights"][neuron][input_index] -= (
                    learning_rate * layer_derivatives["weight_derivatives"][neuron][input_index])

        # bias: one per neuron
        for neuron in range(len(layer["biases"])):
            layer["biases"][neuron] -= learning_rate * \
                layer_derivatives["bias_derivatives"][neuron]


# ========================================================================
# PART 8 - training loop
# ========================================================================
def train_network(network, training, epochs, learning_rate, seed):
    """Train the network. Return the average loss of every epoch."""
    rdm = random.Random(seed)
    average_losses = []

    for epoch in range(epochs):
        # a new order each epoch, so the network doesn't learn the order
        flowers_in_order = list(training)
        rdm.shuffle(flowers_in_order)

        total_loss = 0.0
        for measurements, true_answer in flowers_in_order:
            forward_pass_results = forward_pass(network, measurements)
            total_loss += calculate_loss(
                forward_pass_results["probabilities"], true_answer)

            derivatives = backward_pass(
                network, forward_pass_results, true_answer)
            update_network(network, derivatives, learning_rate)

        average_losses.append(total_loss / len(training))

    return average_losses


# ========================================================================
# PART 9 - testing: forward pass only, count the correct answers
# ========================================================================
def test_network(network, testing):
    """Return the accuracy: the share of test flowers the network gets right."""
    correct = 0
    for measurements, true_answer in testing:
        probabilities = forward_pass(network, measurements)["probabilities"]

        # position of the highest probability
        network_answer = probabilities.index(max(probabilities))
        correct_answer = true_answer.index(1)       # position of the 1

        if network_answer == correct_answer:
            correct += 1

    return correct / len(testing)
