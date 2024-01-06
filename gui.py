import tkinter as tk
import solve

# create the root window
root = tk.Tk()
root.title("Sudoku Solver")

# grid_frame = tk.Frame(root)
# grid_frame.grid(row=0, column=0, padx=10, pady=10)

class GUI():
    def __init__(self, root):
        self.root = root
        self.entries = [[tk.Entry(root, width=2, justify="center") for i in range(9)] for j in range(9)]
        self.create_entries()
        self.create_solve_button()

# Create the grid of entries
entries = [[tk.Entry(root, width=2, justify="center") for i in range(9)] for j in range(9)]

# Place the entries in the grid
for i in range(9):
    for j in range(9):
        entries[i][j].grid(row=i, column=j)

def get_board_values(entries):
    board = []
    for i in range(9):
        row = []
        for j in range(9):
            if entries[i][j].get() == "":
                row.append(0)
            else:
                row.append(int(entries[i][j].get()))
        board.append(row)
    return board

def update_entries(entries, board):
    for i in range(9):
        for j in range(9):
            entries[i][j].delete(0, tk.END)
            entries[i][j].insert(0, board[i][j])

def gui_solve(entries):
    board = get_board_values(entries)
    solve.solve(board)
    update_entries(entries, board)

# Create the solve button
solve_button = tk.Button(root, text="Solve" , command=lambda: gui_solve(entries))
solve_button.grid(row=10, column=0, columnspan=9, padx=10, pady=10)





root.mainloop()