from __future__ import annotations
import enum
from typing import Optional, List
import numpy as np
import random
from termcolor import colored

# Board size
N = 6

# Number of tokens (nodes) each player has to collect, in order
NUM_TOKENS = 5

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

class MoveType(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Node:
    def __init__(self: Node, index: int) -> None:
        self.index = index

        self.row = self.index // N
        self.col = self.index % N
        if not Node._check_valid_position(self.row, self.col):
            raise ValueError("Node index out of bounds.")

    @staticmethod
    def _check_valid_position(row: int, col: int) -> bool:
        return 0 <= row < N and 0 <= col < N

    @staticmethod
    def from_position(row: int, col: int) -> Node:
        if not Node._check_valid_position(row, col):
            raise ValueError("Node position out of bounds.")
        index = row * N + col
        return Node(index) 
    
    def add_move(self: Node, move: MoveType) -> Node:
        delta_dict = {
            MoveType.UP: (-1, 0),
            MoveType.DOWN: (1, 0),
            MoveType.LEFT: (0, -1),
            MoveType.RIGHT: (0, 1)
        }
        delta_row, delta_col = delta_dict[move]
        new_row = self.row + delta_row
        new_col = self.col + delta_col
        return self.from_position(new_row, new_col)
        
    def __eq__(self, other):
        if not isinstance(other, Node):
            raise ValueError("Cannot compare Node with non-Node type.")
        return self.index == other.index
        
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

        self.nodes = np.arange(N**2, dtype=np.int8)
        self.nodes_grid = self.nodes.reshape((N, N))
        self.corner_nodes = [
            Node.from_position(0, 0).index,
            Node.from_position(0, N - 1).index,
            Node.from_position(N - 1, 0).index,
            Node.from_position(N - 1, N - 1).index
        ]

        # This is the starting pos for player 0. 
        # Player 1 will start in the opposite corner.
        self.starting_pos = starting_pos
        self.player_positions = self._init_player_positions()

        # We are going to have a visible adjacency matrix representing players' knowledge of the board,
        # and a hidden adjacency matrix representing the actual board configuration.
        self.visible_adj_matrix = self._init_adj_matrix()
        self.hidden_adj_matrix = self._init_adj_matrix()

        # Each player will have to collect NUM_TOKENS nodes in order
        self.player_tokens = ([], [])
        self.player_tokens = self._init_player_tokens()

        # Randomly remove walls from the hidden adjacency matrix to create the board configuration.
        self.hidden_remove_walls()

        # The die has to be rolled
        self.die_roll = 0
        self._roll_die()

        # Player 0 starts
        self.current_player = 0

    def _roll_die(self: Game) -> None:
        """ Roll the die and update the die_roll attribute. """
        self.die_roll = random.randint(1, 6)

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
        adj_matrix = np.zeros((N**2, N**2), dtype=np.int8)

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
    
    def _player_to_start(self: Game, player: int) -> int:
        """ Return the starting position of the given player. """
        if player == 0:
            return self.corner_nodes[self.starting_pos.value]
        else:
            return self.corner_nodes[3 - self.starting_pos.value]

    def _init_player_positions(self: Game) -> List[int, int]:
        """ Initialize player positions based on the starting position. 
            Player 0 starts at the specified starting position, and Player 1 starts at the opposite corner.
        """
        return [self._player_to_start(0), self._player_to_start(1)]

    def _init_player_tokens(self: Game) -> List[list[int], list[int]]:
        """ Initialize the tokens (nodes) each player has to collect, in order.
            Each player will have NUM_TOKENS unique nodes to collect, randomly selected from the AVAILABLE nodes on the board.
            A node is defined as available if it is not a corner node, nor a neighbor of a corner node in the hidden adjacency matrix.
            The tokens for Player 0 and Player 1 must not overlap.
        """
        # Define unavailable nodes: corner nodes and their neighbors
        unavailable_nodes = set(self.corner_nodes)
        for corner_ind in self.corner_nodes:
            neighbors = np.where(self.hidden_adj_matrix[corner_ind] == 1)[0]
            unavailable_nodes.update(neighbors)

        available_nodes = set(self.nodes) - unavailable_nodes

        # Draw 2 * NUM_TOKENS unique nodes from available nodes
        player_tokens = random.sample(sorted(available_nodes), 2 * NUM_TOKENS)

        # Interleave tokens (player0, player1, player0, ...) for both players to ensure no overlap
        player0_tokens = player_tokens[0::2]
        player1_tokens = player_tokens[1::2]
        return (player0_tokens, player1_tokens)

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
        for node in range(N**2):
            reachable_nodes = self._bfs(node)
            if len(reachable_nodes) != N**2:
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
        output += f"Player 0 still has to collect nodes: {self.player_tokens[0]}\n"
        output += f"Player 1 still has to collect nodes: {self.player_tokens[1]}\n"
        output += f"Player {self.current_player}'s turn. Die roll: {self.die_roll}\n"
        
        player0_pos_idx = self.player_positions[0]
        player1_pos_idx = self.player_positions[1]

        for r in range(N):
            # Print node row
            for c in range(N):
                current_node_idx = Node.from_position(r, c).index
                if current_node_idx == player0_pos_idx:
                    output += colored("●", "red")
                elif current_node_idx == player1_pos_idx:
                    output += colored("●", "blue")
                else:
                    output += "●"
                if c < N - 1:
                    node1_idx = current_node_idx
                    node2_idx = Node.from_position(r, c + 1).index
                    if adj_matrix[node1_idx, node2_idx]:
                        output += "───"
                    else:
                        output += "   "
            output += "\n"

            # Print vertical connections row
            if r < N - 1:
                for c in range(N):
                    node1_idx = Node.from_position(r, c).index
                    node2_idx = Node.from_position(r + 1, c).index
                    if adj_matrix[node1_idx, node2_idx]:
                        output += "│"
                    else:
                        output += " "
                    if c < N - 1:
                        output += "   "
                output += "\n"
        
        print(output)

    def make_move(self: Game, player: int, move: MoveType) -> None:
        """ Make a move for the given player in the specified direction. """
        
        if player != self.current_player:
            print(f"Player {player} attempted to make a move out of turn. Move ignored.")
            raise ValueError("It's not this player's turn.")
        
        if self.die_roll <= 0:
            print(f"Player {player} attempted to make a move with no remaining moves. Move ignored.")
            raise ValueError("The maximum number of moves has already been made.")
        
        start_node_idx = self.player_positions[player]
        end_node_idx = Node(start_node_idx).add_move(move).index

        if self.hidden_adj_matrix[start_node_idx, end_node_idx] == 0:
            # This move is blocked by a wall. 
            # Update the visible adjacency matrix to reflect this knowledge.
            # Bring back the player to their original position, give back control to the next player, 
            # reroll die. 
            print(f"Player {player} hit a wall! The visible map is updated.")
            self._remove_wall(AdjMatrixType.VISIBLE, start_node_idx, end_node_idx)
            self.player_positions[self.current_player] = self._player_to_start(self.current_player)
            self.current_player = 1 - self.current_player
            self._roll_die()
            return
        
        # Update player position
        self.player_positions[self.current_player] = end_node_idx
        self.die_roll -= 1

        # Check if the player has collected their next token
        if end_node_idx in self.player_tokens[player]:
            self.player_tokens[player].remove(end_node_idx)
            print(f"Player {player} collected a token at node {end_node_idx}!")

            if not self.player_tokens[player]:
                self.is_over = True
                print(f"Player {player} has collected all tokens and wins the game!")
                return

        if self.die_roll == 0:
            # The player has used up all their moves. 
            # Switch to the other player's turn and reroll die.
            self.current_player = 1 - self.current_player
            self._roll_die()
            return