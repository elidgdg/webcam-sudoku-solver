import random
import copy

# find first empty cell
def find_empty_cell(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return (row,col)
    return None

# check if placing num at (row, col) is valid
def cell_valid(board, row, col, num):
    # check row and column
    for i in range(9):
        if board[row][i] == num:
            return False
    for i in range(9):
        if board[i][col] == num:
            return False
    # check box
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def solve(board):
    # if no empty cell, Sudoku is solved
    if find_empty_cell(board) == None:
        return True
    
    # find next empty cell
    row, col = find_empty_cell(board)

    # try numbers from 1 to 9 in empty cell
    for num in range(1,10):
        if cell_valid(board, row, col, num):
            # if valid, place num in empty cell
            board[row][col] = num

            # recursively solve the rest of the board
            if solve(board):
                return True
            
            # if current digit does not lead to solution, backtrack
            board[row][col] = 0

    # if no digit from 1 to 9 leads to a solution, return False
    return False

# check if board is valid
def is_valid_board(board):
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != 0:
                board[i][j] = 0 # temporarily set cell to 0
                if not cell_valid(board, i, j, num):
                    return False
                board[i][j] = num # reset cell to original value
    return True

def random_hint(board):
    # get current board
    board = copy.deepcopy(board)
    # get solution
    solution = copy.deepcopy(board)
    solve(solution)

    # find empty cells
    empty_cells = []
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                empty_cells.append((i,j))
    
    if len(empty_cells) == 0:
        return None
    
    # choose random empty cell
    row, col = random.choice(empty_cells)

    # set cell to solution
    board[row][col] = solution[row][col]

    return board

    
