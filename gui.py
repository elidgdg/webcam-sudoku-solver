import tkinter as tk
import solve

# grid_frame = tk.Frame(root)
# grid_frame.grid(row=0, column=0, padx=10, pady=10)

class GUI():
    def __init__(self, root):
        self.root = root

        self.entries = [[tk.Entry(root, width=3, justify="center") for i in range(9)] for j in range(9)]
        for i in range(9):
            for j in range(9):

                padx = (0, 0)
                pady = (0, 0)
                if i % 3 == 0 and i != 0:
                    pady = (5, 0)
                if j % 3 == 0 and j != 0:
                    padx = (5, 0)

                self.entries[i][j].grid(row=i, column=j, padx=padx, pady=pady, ipadx=5, ipady=5)

        self.solve_button = self.create_solve_button()
    
    def get_board_values(self):
        board = []
        for i in range(9):
            row = []
            for j in range(9):
                if self.entries[i][j].get() == "":
                    row.append(0)
                else:
                    row.append(int(self.entries[i][j].get()))
            board.append(row)
        return board
    
    def update_entries(self, board):
        for i in range(9):
            for j in range(9):
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, board[i][j])

    def gui_solve(self):
        board = self.get_board_values()
        solve.solve(board)
        self.update_entries(board)
    
    def create_solve_button(self):
        solve_button = tk.Button(self.root, text="Solve" , command=lambda: self.gui_solve())
        solve_button.grid(row=10, column=0, columnspan=9, padx=10, pady=10)
        return solve_button