# # Main Author: Tanvir Singh
# # Main Reviewer: MeetSiamr, Kashyap
from Queue import Queue
import copy


# This functions Returns the coordinates of the Cells that are overflowing
# grid = [
# [1,2,3,4]
# [2,3,4,5]
# It is Rectangular, no checks for required***
# ]
def get_overflow_list(grid):
    """
    Determine the cells in the grid that are overflowing.

    :param grid: A 2D list of cells.
    :return: List of (row, col) coordinates of overflowing cells or None if none are overflowing.
    """

    # We need to check the size of our grid [rows, columns]
    row_len = len(grid)
    column_len = len(grid[0])

    # We need to Check whether our Cell is Edge, corner, or not on the outside
    def cell_positioning(row, col):
        # check if edges.
        if (row, col) in [(0, 0), (0, column_len - 1), (row_len - 1, 0), (row_len - 1, column_len - 1)]:
            return "corner"
            # Check if the cell is on an edge but not a corner
        elif row in [0, row_len - 1] or col in [0, column_len - 1]:
            return "edge"
            # If it's neither a corner nor an edge, it's an inner cell
        else:
            return "inner"

    # Based on Cell Position we need to make sure if a cell overflows
    overflow = []
    for r in range(row_len):
        for c in range(column_len):
            cell_type = cell_positioning(r, c)
            if (cell_type == "corner" and abs(grid[r][c]) >= 2) or \
                    (cell_type == "edge" and abs(grid[r][c]) >= 3) or \
                    (cell_type == "inner" and abs(grid[r][c]) >= 4):
                overflow.append((r, c))  # appending the coordinates

    # if overflow list is empty, return None, else return the list
    return None if not overflow else overflow


def overflow(grid, a_queue):
    """
    Simulate the overflow process and enqueue each grid state into the queue.

    :param grid: Initial 2D list of cells.
    :param a_queue: Queue to store each state of the grid during the overflow process.
    :return: Number of iterations until no cells are overflowing.
    """
    overflow_cells = get_overflow_list(grid)
    row_len = len(grid)
    col_len = len(grid[0])

    # Base case: if there are no overflowing cells, we're done
    if overflow_cells is None:
        return 0

    new_grid = copy.deepcopy(grid)
    # First, let's set all overflowing cells to 0
    for cell in overflow_cells:
        r, c = cell
        new_grid[r][c] = 0

    # setting neighbors
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Directions: Up, Down, Left, Right
    for cell in overflow_cells:
        r, c = cell
        sign = grid[r][c] // abs(grid[r][c])
        for dr, dc in dirs:
            if 0 <= r + dr < row_len and 0 <= c + dc < col_len:
                new_grid[r + dr][c + dc] = (abs(new_grid[r + dr][c + dc]) * sign) + (1 * sign)

    value = 0
    negvalue = 0

    for r in range(row_len):
        for c in range(col_len):
            if new_grid[r][c] >= 0:
                value += 1
            elif new_grid[r][c] <= 0:
                negvalue += 1

    # Check if all values are positive or all values are negative
    if value == row_len * col_len or negvalue == row_len * col_len:
        a_queue.enqueue(new_grid)
        return 1

    # Continue with the rest of your code if the condition is not met

    a_queue.enqueue(new_grid)

    # Recursive call
    return 1 + overflow(new_grid, a_queue)


def copy_board(board):
    """
    Create a deep copy of the given 2D board.

    :param board: The original game board.
    :return: A new 2D list representing a deep copy of the original board.
    """
    current_board = []
    height = len(board)
    for i in range(height):
        current_board.append(board[i].copy())
    return current_board


def evaluate_board(board, player):
    """
    Evaluate the game board based on the given player's perspective.

    :param board: The current game board.
    :param player: The player for whom the board is being evaluated (1 or -1).
    :return: The evaluation score of the board for the specified player.
    """
    score = 0
    opponentscore = 0
    WIN_SCORE = 1000
    LoosingScore = -1000
    row_len = len(board)
    col_len = len(board[0])

    for r in range(row_len):
        for c in range(col_len):
            piece = board[r][c]
            if piece != 0:
                if (player == 1 and piece > 0) or (player == -1 and piece < 0):
                    score += abs(piece)
                else:
                    opponentscore += abs(piece)

    if opponentscore != 0 and score != 0:
        return (score - opponentscore) * player
    elif score == 0 and opponentscore != 0:
        return LoosingScore
    elif opponentscore == 0 and score != 0:
        return WIN_SCORE


from a1_partc import Queue


class GameTree:
    """
    Class representing a game tree for a board game.

    Attributes:
    - root (Node): The root node of the game tree.
    """

    class Node:
        """
        Initialize a node in the game tree.

        Parameters:
        - board (list): The game board associated with the node.
        - depth (int): The depth of the node in the game tree.
        - player (int): The player associated with the node.
        - move (tuple): The move that led to the current board state.
        - tree_height (int): The maximum height of the game tree.
        """

        def __init__(self, board, depth, player, move, tree_height=4):
            self.board = copy_board(board)
            self.depth = depth
            self.player = player
            self.tree_height = tree_height
            self.children = []
            self.score = None
            self.move = move

    def __init__(self, board, player, tree_height=4):
        """
        Initialize the game tree with the root node.

        Parameters:
        - board (list): The initial game board.
        - player (int): The player for whom the game tree is constructed.
        - tree_height (int): The maximum height of the game tree.
        """
        self.player = player
        self.root = self.Node(board, 0, player, (0, 4), tree_height)
        self.build_tree(self.root)

    def build_tree(self, node):
        """
        Recursively build the game tree.

        Parameters:
        - node (Node): The current node in the game tree.
        """
        samecolor = self.same_color_tree(node.board)
        if node.depth < self.root.tree_height - 1 and not samecolor:
            child_boards, moves = self.generate_child(node.board, node.player)
            node.children = [self.Node(child_board, node.depth + 1, -node.player, move, node.tree_height) for
                             child_board, move in zip(child_boards, moves)]

            for child_node in node.children:
                self.build_tree(child_node)

    def generate_child(self, board, player):
        """
        Generate child nodes and corresponding moves based on the current board state.

        Parameters:
        - board (list): The current game board.
        - player (int): The player for whom child nodes are generated.

        Returns:
        - tuple: A tuple containing lists of child boards and corresponding moves.
        """
        row_len = len(board)
        col_len = len(board[0])
        child_boards = []
        state_queue = Queue()
        moves = []

        for r in range(row_len):
            for c in range(col_len):
                if (board[r][c] >= 0 and player == 1) or (board[r][c] <= 0 and player == -1):
                    new_board = copy_board(board)
                    move = (r, c)
                    new_board[r][c] += player  # Adjust based on the player
                    moves.append(move)

                    rc = overflow(new_board, state_queue)  # Check and handle overflow
                    if rc != 0:
                        for i in range(rc):
                            tmp = state_queue.dequeue()
                        child_boards.append(tmp)
                    else:
                        child_boards.append(new_board)

        return child_boards, moves

    def get_move(self):
        """
        Find and return the best move using the minimax algorithm.

        Returns:
        - tuple: The best move for the current player.
        """
        # Start the minimax algorithm
        best_score = float('-inf') if self.player == 1 else float('inf')
        best_move = None
        for child in self.root.children:
            # For each child node, calculate the minimax score
            # if child.player == 1:
            #  score = self.minimax(child, True)
            # else:
            if (self.player == 1):
                score = self.minimax(child, False)
            else:
                score = self.minimax(child, True)

            # Update the best move if the current score is better
            if (self.player == 1 and score > best_score) or (self.player == -1 and score < best_score):
                best_score = score
                best_move = child.move

        return best_move

    def minimax(self, node, maximizing_player, alpha=float('-inf'), beta=float('inf')):
        """
        Implement the minimax algorithm to find the optimal move.

        Parameters:
        - node (Node): The current node being evaluated.
        - maximizing_player (bool): True if the current player is maximizing, False otherwise.
        - alpha (float): The alpha value for alpha-beta pruning.
        - beta (float): The beta value for alpha-beta pruning.

        Returns:
        - int: The minimax score for the current node.
        """
        if not node.children:
            score = evaluate_board(node.board, node.player if maximizing_player else -node.player)
            return score

        if maximizing_player:
            max_eval = float('-inf')
            for child in node.children:
                eval = self.minimax(child, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                eval = self.minimax(child, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def same_color_tree(self, board):
        return not (any(piece > 0 for row in board for piece in row) and any(
            piece < 0 for row in board for piece in row))

    def clear_tree(self):
        self.root = None

    def __del__(self):
        self.clear_tree()


