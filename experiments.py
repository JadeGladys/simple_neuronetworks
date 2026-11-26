import time
import csv
from neuro_networks_algorithm import (
    load_flowers, prepare_flowers, create_network, train_network, test_network
)

# SETTINGS

DATA_FILE = "iris.data"
BASE_SEED = 42
NUMBER_OF_RUNS = 5       # how many runs per setting (one seed each)
START_RANGE = 0.5       # starting weights and biases are random between -0.5 and 0.5

# the values to try: low, medium, high
LEARNING_RATES = [0.001, 0.01, 0.1]   # step size
# how many times we go through all the training flowers
EPOCHS = [50, 100, 200]
# neurons in EACH hidden layer
HIDDEN_NEURONS = [3, 10, 20]
# share of flowers used for training (the rest for testing)
TRAIN_SHARES = [0.5, 0.6, 0.8]


# ========================================================================
# PART 1 - run one experiment
# ========================================================================
def run_experiment(flowers, hidden_neurons, learning_rate, epochs, train_share, seed):
    """Prepare, create, train and test one network. Return its results."""
    start_time = time.perf_counter()

    training, testing = prepare_flowers(flowers, train_share, seed)
    network = create_network(hidden_neurons, hidden_neurons, START_RANGE, seed)
    average_losses = train_network(
        network, training, epochs, learning_rate, seed)
    accuracy = test_network(network, testing)

    seconds = time.perf_counter() - start_time

    return {
        "final_loss": average_losses[-1],    # average loss of the last epoch
        "accuracy": accuracy,
        "time": seconds,
        "average_losses": average_losses,    # one per epoch, for the loss graph
    }


# ========================================================================
# PART 2 - run every combination, a number of times each per seed
# ========================================================================
def run_all_experiments(flowers):
    """Return (results rows, loss curve rows), ready to save as CSV."""
    results_rows = []
    loss_curve_rows = []

    total_combinations = (len(LEARNING_RATES) * len(EPOCHS)
                          * len(HIDDEN_NEURONS) * len(TRAIN_SHARES))
    combination_number = 0

    for learning_rate in LEARNING_RATES:
        for epochs in EPOCHS:
            for hidden_neurons in HIDDEN_NEURONS:
                for train_share in TRAIN_SHARES:
                    combination_number += 1

                    accuracies = []
                    final_losses = []
                    time_list = []
                    summed_average_losses = [0.0] * epochs

                    # run this combination a number of times with different seeds
                    for run_number in range(NUMBER_OF_RUNS):
                        seed = BASE_SEED + run_number

                        result = run_experiment(
                            flowers, hidden_neurons, learning_rate, epochs, train_share, seed)

                        accuracies.append(result["accuracy"])
                        final_losses.append(result["final_loss"])
                        time_list.append(result["time"])
                        for epoch in range(epochs):
                            summed_average_losses[epoch] += result["average_losses"][epoch]

                    average_accuracy = sum(accuracies) / NUMBER_OF_RUNS
                    average_final_loss = sum(final_losses) / NUMBER_OF_RUNS
                    average_seconds = sum(time_list) / NUMBER_OF_RUNS

                    # one row per combination
                    results_rows.append([
                        learning_rate, epochs, hidden_neurons, train_share,
                        round(average_accuracy, 4),
                        round(min(accuracies), 4),   # worst of the 5 runs
                        round(max(accuracies), 4),   # best of the 5 runs
                        round(average_final_loss, 4),
                        round(average_seconds, 3),
                    ])

                    # one row per epoch, for the loss graphs
                    for epoch in range(epochs):
                        loss_curve_rows.append([
                            learning_rate, epochs, hidden_neurons, train_share,
                            epoch + 1,
                            round(
                                summed_average_losses[epoch] / NUMBER_OF_RUNS, 4),
                        ])

                    print(f"{combination_number}/{total_combinations}  "
                          f"rate={learning_rate}  epochs={epochs}  neurons={hidden_neurons}  "
                          f"train={train_share}  ->  accuracy={average_accuracy:.3f}  "
                          f"time={average_seconds:.2f}s")

    return results_rows, loss_curve_rows


# ========================================================================
# PART 3 - save to CSV
# ========================================================================
def save_csv(file_name, header, rows):
    """Write a header row and all the rows to a CSV file."""
    with open(file_name, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    """Run all the experiments, and save it to a CSV file."""
    flowers = load_flowers(DATA_FILE)
    results_rows, loss_curve_rows = run_all_experiments(flowers)

    save_csv("results.csv",
             ["learning_rate", "epochs", "hidden_neurons", "train_share",
              "average_accuracy", "lowest_accuracy", "highest_accuracy",
              "average_final_loss", "average_seconds"],
             results_rows)

    save_csv("loss_curves.csv",
             ["learning_rate", "epochs", "hidden_neurons", "train_share",
              "epoch", "average_loss"],
             loss_curve_rows)

    print("\nSaved results.csv and loss_curves.csv")


if __name__ == "__main__":
    main()
