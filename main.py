import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


def matrix(matrix_, addition):
    try:
        return np.linalg.solve(matrix_, addition).T  # This one took 2 minutes to figure out
    except Exception as e:
        return e


class LiveMatrix():
    def __init__(self, matr, generations, matrix_size=tuple(), generate_matrix=False):
        self.generations = generations
        self.matr = matr
        self.generate_matrix = generate_matrix
        self.matrix_size = matrix_size

    def matgen(self):
        # Function for generating random matrices
        # We use try in case passing wrong matrix size
        try:
            return np.random.randint(2, size=self.matrix_size)
        except Exception as e:
            raise e

    def run_through(self, visualize=True, verbose=1):
        # Check if we want to generate random matrix
        if self.generate_matrix:
            matr = self.matgen()
        else:
            matr = self.matr
            self.matrix_size = matr.shape

        # Collecting stats about each generation
        cells_alive_stats = {}
        matrices = np.zeros((self.generations + 1, self.matrix_size[0], self.matrix_size[1]))

        # 0 generation is matrix which was given as argument
        matrix_cells_alive = np.sum(matr)
        cells_alive_stats[0] = matrix_cells_alive  # Amount of alive cells over generation
        matrices[0] = matr  # Array with each matrix generation

        for generation in range(1, self.generations + 1):
            # We create padding for easier cell neighbors access
            matr_with_padding = np.pad(matr, (1, 1), mode='constant', constant_values=0)
            new_matrix_generation = np.zeros_like(matr)  # Mask of next generation

            for i in range(1, len(matr_with_padding) - 1):  # We start at 1 and finish at -1 because of padding
                for j in range(1, len(matr_with_padding) - 1):
                    start_row = i - 1
                    start_col = j - 1
                    cell = matr_with_padding[i][j]  # Our cell value

                    # We retrieve 3x3 neighbors matrix from our main matrix
                    neighbors = matr_with_padding[start_row:start_row + 3, start_col:start_col + 3]

                    # Sum of all cells alive in neighbors matrix
                    cells_alive = np.sum(neighbors)
                    if cell == 0 and cells_alive == 3:
                        new_matrix_generation[i - 1][j - 1] = 1
                    if cell == 1 and cells_alive in (3, 4):
                        new_matrix_generation[i - 1][j - 1] = 1

            # Collect data from this iteration
            matrix_cells_alive = np.sum(new_matrix_generation)
            cells_alive_stats[generation] = matrix_cells_alive
            matrices[generation] = new_matrix_generation

            # Pass new matrix to next iterarion
            matr = new_matrix_generation

            if verbose == 1:
                print(f'Current matrix: \n{new_matrix_generation}')
                print(f'Cells alive: {matrix_cells_alive}')
                print(f'Generation: {generation}')
                print('\n')

        if visualize:
            # Sum of all matrices to see how amount of generations in which each sell survived
            cells_heatmap = np.zeros_like(matrices[0])
            for gen in matrices:
                cells_heatmap = np.add(cells_heatmap, gen)

            # Plot each generation in one window
            fig, ax = plt.subplots(1, self.generations + 1)
            for gen in range(self.generations + 1):
                sn.heatmap(matrices[gen], square=True, cmap='viridis', cbar=False, annot=False, ax=ax[gen]).set(
                    title=f'Generation {gen}')
            plt.show()

            # Plot heatmap
            sn.heatmap(cells_heatmap, square=True, cmap='viridis', annot=True).set(
                title=f'Cells alive over all generations')
            plt.show()
            # Plot alive cells stats over each generation
            sn.lineplot(cells_alive_stats).set(title='All cells alive over generation')
            plt.show()

        print('Task 2\n', matrices[-1], '\n')


def prob_count(coin_data: list, flip_data: list, verbose=False):

    # Add equal chances to be picked for every coin
    prob_data = {i: 1/len(coin_data) for i in coin_data}

    # Find probability of heads for basic case
    heads_prob = sum([i * prob_data[i] for i in prob_data])
    results = {}
    for x, flip in enumerate(flip_data):
        for coin in prob_data:
            if flip == 1:
                prob_data[coin] = coin * prob_data[coin] / heads_prob  # Using Bayes' theorem for each coin
            elif flip == 0:
                prob_data[coin] = ((1 - coin) * prob_data[coin]) / (1 - heads_prob)  # Same theorem, but for tails

        heads_prob = sum([i * prob_data[i] for i in prob_data])  # Finding probability of getting heads
        results[x] = heads_prob  # Add to results

        if verbose:  # In case we need to change something / tune program
            print(f'flip: {flip}')
            print(f'Heads probability: {heads_prob}')
            print(prob_data)
            print('\n')

    print('Task3\n', f'Heads probability: {results}')


if __name__ == '__main__':

    # Task 1
    matrix_1 = np.array([[1, 2, 3], [0, 1, 2], [2, 0, 0]])
    add_1 = np.array([[1], [1], [0]])

    print('Task1\n', matrix(matrix_1, add_1), '\n')

    # Task 2
    test_matrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 0, 0, 1],
                            [0, 1, 1, 0, 1, 1, 0],
                            [1, 1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1, 0, 1]])

    matrix = LiveMatrix(test_matrix, 7)
    matrix.run_through(verbose=0, visualize=False)

    # Task 2.1-2.2**
    matrix = LiveMatrix(None, 7, (10, 10), generate_matrix=True)
    matrix.run_through(verbose=0)

    # Task 3
    coins = [0.1, 0.2, 0.4, 0.8, 0.9]
    flips = [1, 1, 1, 0, 1, 0, 1, 1]

    prob_count(coins, flips, verbose=False)
