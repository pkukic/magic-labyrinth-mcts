from Game import Game

def main():
    init_walls = 24
    g = Game(walls=init_walls)
    print("Initial board:")
    g.visualize_adj_matrix()
    g.remove_walls()
    print(f"Board after removing {init_walls} walls:")
    g.visualize_adj_matrix()

if __name__ == "__main__":
    main()
