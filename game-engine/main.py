from Game import Game
from Game import AdjMatrixType

def main():
    init_walls = 24
    g = Game(walls=init_walls)
    print(f"Hidden board after removing {init_walls} walls:")
    g.visualize_adj_matrix(AdjMatrixType.HIDDEN)
    print(f"Visible board (no walls removed):")
    g.visualize_adj_matrix(AdjMatrixType.VISIBLE)

if __name__ == "__main__":
    main()
