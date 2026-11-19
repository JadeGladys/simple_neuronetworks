import random

# SETTINGS - numbers you choose yourself
DATA_FILE = "iris.data"
TRAIN_SHARE = 0.6   # 60% of flowers for training, 40% for testing
SEED = 42           # same seed = same random numbers every run (fair tests)


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
                measurements.append(num)

            species = parts[4]
            flowers.append((measurements, species))
    return flowers


# =====================================================================
# PART 2 - prepare the data (shuffle, split into training/testing)
# =====================================================================
SPECIES_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Turn a species name into its true answer list, e.g. "versicolor" -> [0, 1, 0].


def species_to_true_answer(species):
    """Return [1, 0, 0], [0, 1, 0] or [0, 0, 1]."""
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
