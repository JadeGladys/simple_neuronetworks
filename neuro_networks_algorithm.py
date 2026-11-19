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
