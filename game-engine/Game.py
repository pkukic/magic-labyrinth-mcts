from __future__ import annotations
import enum
from typing import Optional, Tuple
import numpy as np
import random
from termcolor import colored

# Board size
N = 6

# Number of nodes in a row of the mesh
NODES_ROW = N + 1

# Number of nodes in the mesh
NUM_NODES = NODES_ROW ** 2

# Minimum number of walls on the board
MIN_WALLS = 19

# Maximum number of walls on the board
MAX_WALLS = 24

class AdjMatrixType(enum.Enum):
    VISIBLE = 1
    HIDDEN = 2

class StartingPos(enum.Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3

class Game:
    def __init__(self: Game, starting_pos: Optional[StartingPos] = StartingPos.TOP_LEFT, walls: Optional[int] = MIN_WALLS) -> None:
        """ Initialize the game state. 
        
        Args:
            starting_player (int, 0 or 1): The player who will make the first move. There are two possible players. 
            walls (int): The number of walls (edges in graph) present on the board. It can range between 19 and 24.
        """

        # Ensure valid walls parameter
        # Since these are external inputs, we will throw exceptions for invalid values.
        if not (MIN_WALLS <= walls <= MAX_WALLS):
            raise ValueError(f"walls must be between {MIN_WALLS} and {MAX_WALLS}.")

        # Player 0 always starts first. 
        # You will be able to choose whether you start as player 0 or player 1 in the MCTS agent.
        self.current_player = 0
        self.is_over = False
        self.walls = walls

        self.nodes_grid = np.arange(NUM_NODES).reshape((N + 1, N + 1)) 
        self.corner_nodes = [
            self._flatten_node(0, 0),
            self._flatten_node(0, N),
            self._flatten_node(N, 0),
            self._flatten_node(N, N)
        ]

        # This is the starting pos for player 0. 
        # Player 1 will start in the opposite corner.
        self.starting_pos = starting_pos
        self.player_positions = self._init_player_positions()

        # We are going to have a visible adjacency matrix representing players' knowledge of the board,
        # and a hidden adjacency matrix representing the actual board configuration.
        self.visible_adj_matrix = self._init_adj_matrix()
        self.hidden_adj_matrix = self._init_adj_matrix()

        # Randomly remove walls from the hidden adjacency matrix to create the board configuration.
        self.hidden_remove_walls()

    def _init_adj_matrix(self: Game) -> np.ndarray:
        """ Initialize the adjacency matrix. 
            The board has N x N squares, which can be represented by a mesh graph of size (N + 1) x (N + 1) nodes.
            The graph will be represented by an adjacency matrix of size (N + 1)^2 x (N + 1)^2. 
            The nodes are numbered from 0 to (N + 1)^2 - 1, starting from the top-left corner and moving row-wise to the bottom-right corner.
            The adjacency matrix has to be such that:
                1. Each node is connected to its adjacent nodes (up, down, left, right).
                2. The edges are bidirectional.
                3. The edges on the boundary of the graph are not connected to any other nodes outside the graph.
        """
        # Horizontal connections
        adj_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.int8)

        h_left = self.nodes_grid[:, :-1].flatten()
        h_right = self.nodes_grid[:, 1:].flatten()
        adj_matrix[h_left, h_right] = 1
        adj_matrix[h_right, h_left] = 1

        # Vertical connections
        v_up = self.nodes_grid[:-1, :].flatten()
        v_down = self.nodes_grid[1:, :].flatten()
        adj_matrix[v_up, v_down] = 1
        adj_matrix[v_down, v_up] = 1

        return adj_matrix
    
    def _init_player_positions(self: Game) -> Tuple[Tuple[int, int]]:
        """ Initialize player positions based on the starting position. 
            Player 0 starts at the specified starting position, and Player 1 starts at the opposite corner.
        """
        if self.starting_pos == StartingPos.TOP_LEFT:
            return ((0, 0), (N, N))
        elif self.starting_pos == StartingPos.TOP_RIGHT:
            return ((0, N), (N, 0))
        elif self.starting_pos == StartingPos.BOTTOM_LEFT:
            return ((N, 0), (0, N))
        else:
            return ((N, N), (0, 0))

    def _flatten_node(self: Game, node_row: int, node_col: int) -> int:
        """ Convert a node's (row, col) position in the grid to its corresponding index in the adjacency matrix. """
        if not (0 <= node_row < (N + 1)) or not (0 <= node_col < (N + 1)):
            raise ValueError("Node position out of bounds.")
        return node_row * (N + 1) + node_col

    def _bfs(self: Game, start_node: int) -> set[int]:
        """ Perform a breadth-first search (BFS) to find all nodes reachable from the start_node. """
        visited = set()
        queue = [start_node]
        visited.add(start_node)

        while queue:
            current_node = queue.pop(0)
            neighbors = np.where(self.hidden_adj_matrix[current_node] == 1)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited

    def _check_board_legal(self: Game) -> bool:
        """ Check if the current board configuration is legal. 
            The board configuration is legal if the graph is connected (i.e., there is a path between any two nodes).
        """
        for node in range(NUM_NODES):
            reachable_nodes = self._bfs(node)
            if len(reachable_nodes) != NUM_NODES:
                return False
        return True

    def _remove_wall(self: Game, adj_matrix_type: AdjMatrixType, start_node: int, end_node: int) -> None:
        """ Remove a wall (edge) between two nodes in the adjacency matrix. """
        adj_matrix = self.hidden_adj_matrix
        if adj_matrix_type == AdjMatrixType.VISIBLE:
            adj_matrix = self.visible_adj_matrix

        adj_matrix[start_node, end_node] = 0
        adj_matrix[end_node, start_node] = 0

    def _restore_wall(self: Game, adj_matrix_type: AdjMatrixType, start_node: int, end_node: int) -> None:
        """ Restore a wall (edge) between two nodes in the adjacency matrix. """
        adj_matrix = self.hidden_adj_matrix
        if adj_matrix_type == AdjMatrixType.VISIBLE:
            adj_matrix = self.visible_adj_matrix
        adj_matrix[start_node, end_node] = 1
        adj_matrix[end_node, start_node] = 1   
    
    def hidden_remove_walls(self: Game) -> None:
        """ Randomly remove walls (edges) from the adjacency matrix until the desired number of walls (self.walls) is removed.
            This function ensures that the board remains legal after each wall removal.
        """
        
        # Find all existing edges (walls)
        rows, cols = np.where(np.triu(self.hidden_adj_matrix, k=1))
        possible_walls_to_remove = list(zip(rows, cols))
        random.shuffle(possible_walls_to_remove)

        walls_removed = 0
        walls_to_remove = self.walls

        if walls_to_remove <= 0:
            return

        while walls_removed < walls_to_remove and possible_walls_to_remove:
            start_node, end_node = possible_walls_to_remove.pop()

            # Temporarily remove the wall
            self._remove_wall(AdjMatrixType.HIDDEN, start_node, end_node)

            # If the board is no longer legal, restore the wall
            if not self._check_board_legal():
                self._restore_wall(AdjMatrixType.HIDDEN, start_node, end_node)
            else:
                walls_removed += 1

    def visualize_adj_matrix(self: Game, type: AdjMatrixType) -> None:
        """ Visualize the adjacency matrix. 
            Nodes are represented as UTF-8 circles (●), and edges are represented as lines (─, │).
            Squares are represented as spaces.
            The position of Player 0 is represented as a red circle (●), and the position of Player 1 is represented as a blue circle (●).
        """

        adj_matrix = self.hidden_adj_matrix
        if type == AdjMatrixType.VISIBLE:
            adj_matrix = self.visible_adj_matrix

        # Construct everything as a string, and then print it at once
        # This is faster. 
        output = ""
        for r in range(N + 1):
            # Print node row
            for c in range(N + 1):
                if (r, c) == self.player_positions[0]:
                    output += colored("●", "red")
                elif (r, c) == self.player_positions[1]:
                    output += colored("●", "blue")
                else:
                    output += "●"
                if c < N:
                    node1 = self._flatten_node(r, c)
                    node2 = self._flatten_node(r, c + 1)
                    if adj_matrix[node1, node2]:
                        output += "───"
                    else:
                        output += "   "
            output += "\n"

            # Print vertical connections row
            if r < N:
                for c in range(N + 1):
                    node1 = self._flatten_node(r, c)
                    node2 = self._flatten_node(r + 1, c)
                    if adj_matrix[node1, node2]:
                        output += "│"
                    else:
                        output += " "
                    if c < N:
                        output += "   "
                output += "\n"
        
        print(output)

