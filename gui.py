import tkinter as tk
import sudoku_algorithms

# grid_frame = tk.Frame(root)
# grid_frame.grid(row=0, column=0, padx=10, pady=10)

class GUI():
    def __init__(self, root):
        self.root = root

        # Create frame for the sudoku grid
        self.grid_frame = tk.Frame(root)
        self.grid_frame.grid(row=0, column=0, padx=10, pady=10)
        # create frame for buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=0, column=1, padx=10, pady=10)

        # Create the 9x9 grid of entry boxes
        self.entries = [[tk.Entry(self.grid_frame, width=3, justify="center") for i in range(9)] for j in range(9)]
        for i in range(9):
            for j in range(9):

                # Add padding to the entry boxes to make them look like a sudoku grid
                padx = (0, 0)
                pady = (0, 0)
                if i % 3 == 0 and i != 0:
                    pady = (5, 0)
                if j % 3 == 0 and j != 0:
                    padx = (5, 0)

                # Add the entry box to the grid
                self.entries[i][j].grid(row=i, column=j, padx=padx, pady=pady, ipadx=5, ipady=5)

        # Create instructions label
        self.instructions = tk.Label(self.button_frame, text="Enter known digits, then click 'Start Solving'", wraplength=90, justify="center")
        self.instructions.grid(row=0, column=0, columnspan=9, padx=10, pady=10) 

        # Create initial buttons
        self.start_solving_btn = tk.Button(self.button_frame, text="Start Solving", command=self.start_solving)
        self.start_solving_btn.grid(row=1, column=0, columnspan=9, padx=10, pady=10)
        self.camera_btn = tk.Button(self.button_frame, text="Camera", command=self.open_camera)
        self.camera_btn.grid(row=2, column=0, columnspan=9, padx=10, pady=10)
        
    def start_solving(self):
        if not sudoku_algorithms.is_valid_board(self.get_board_values()):
            print("Invalid board")
            print(self.root.winfo_width())
            print(self.root.winfo_height())
            return
        # Destroy initial buttons
        self.start_solving_btn.destroy()
        self.camera_btn.destroy()

        # Make any non-empty entries read-only
        for i in range(9):
            for j in range(9):
                if self.entries[i][j].get() != "":
                    self.entries[i][j].config(state="readonly")

        # Create solve, reset, and new puzzle buttons
        self.see_solution_btn = tk.Button(self.button_frame, text="See Solution", command=self.see_solution)
        self.see_solution_btn.grid(row=0, column=0, columnspan=9, padx=10, pady=10)
        self.reset_btn = tk.Button(self.button_frame, text="Reset", command=self.reset)
        self.reset_btn.grid(row=1, column=0, columnspan=9, padx=10, pady=10)
        self.new_puzzle_btn = tk.Button(self.button_frame, text="New Puzzle", command=self.new_puzzle)
        self.new_puzzle_btn.grid(row=2, column=0, columnspan=9, padx=10, pady=10)

    def see_solution(self):
        board = self.get_board_values()
        sudoku_algorithms.solve(board)
        self.update_entries(board)
    
    def reset(self):
        pass
    def new_puzzle(self):
        pass
    def open_camera(self):
        pass


    def get_board_values(self):
        board = []
        # get values from entry boxes
        for i in range(9):
            row = []
            for j in range(9):
                # if entry box is empty, set value to 0
                if self.entries[i][j].get() == "":
                    row.append(0)
                else:
                    row.append(int(self.entries[i][j].get()))
            board.append(row)
        return board
    
    def update_entries(self, board):
        # update entry boxes with values from board
        for i in range(9):
            for j in range(9):
                self.entries[i][j].delete(0, tk.END) # delete current value
                self.entries[i][j].insert(0, board[i][j]) # insert new value
    
    def create_solve_button(self):
        solve_button = tk.Button(self.button_frame, text="Solve" , command=lambda: self.gui_solve())
        solve_button.grid(row=10, column=0, columnspan=9, padx=10, pady=10)
        return solve_button