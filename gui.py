import tkinter as tk
import sudoku_algorithms
import copy
import cv2
from PIL import Image, ImageTk
import image_processing


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

                # Add a callback to the entry box to only allow digits
                reg = self.entries[i][j].register(self.callback)
                self.entries[i][j].config(validate="key", validatecommand=(reg, "%P"))


        # Create message label
        self.message_lbl = tk.Label(self.button_frame, text="Enter known digits, then click 'Start Solving'", wraplength=90, justify="center")
        self.message_lbl.grid(row=0, column=0, columnspan=9, padx=10, pady=10) 

        # Create initial buttons
        self.start_solving_btn = tk.Button(self.button_frame, text="Start Solving", command=self.start_solving)
        self.start_solving_btn.grid(row=1, column=0, columnspan=9, padx=10, pady=10)
        self.camera_btn = tk.Button(self.button_frame, text="Camera", command=self.open_camera)
        self.camera_btn.grid(row=2, column=0, columnspan=9, padx=10, pady=10)

        # Variable for intial entries
        self.initial_entries = []

    def highlight_conflicts(self, row, col):
        board = self.get_board_values()
        # check row
        for i in range(9):
            if board[row][i] == 0:
                self.entries[row][i].config(bg="white")
            else:
                temp = board[row][i]
                board [row][i] = 0
                if sudoku_algorithms.cell_valid(board, row, i, temp) == False:
                    self.entries[row][i].config(bg="red")
                else:
                    self.entries[row][i].config(bg="white")
                board[row][i] = temp

        # check column
        for i in range(9):
            if board[i][col] == 0:
                self.entries[i][col].config(bg="white")
            else:
                temp = board[i][col]
                board [i][col] = 0
                if sudoku_algorithms.cell_valid(board, i, col, temp) == False:
                    self.entries[i][col].config(bg="red")
                else:
                    self.entries[i][col].config(bg="white")
                board[i][col] = temp
        
        # check box
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == 0:
                    self.entries[start_row + i][start_col + j].config(bg="white")
                else:
                    temp = board[start_row + i][start_col + j]
                    board[start_row + i][start_col + j] = 0
                    if sudoku_algorithms.cell_valid(board, start_row + i, start_col + j, temp) == False:
                        self.entries[start_row + i][start_col + j].config(bg="red")
                    else:
                        self.entries[start_row + i][start_col + j].config(bg="white")
                    board[start_row + i][start_col + j] = temp
        
    def callback(self, input):
        if len(input) <= 1 and input.isdigit() or input == "":
            # highlight conflicts
            self.root.after(10, lambda: self.highlight_conflicts(int(self.root.focus_get().grid_info()["row"]), int(self.root.focus_get().grid_info()["column"])))
            return True
        else:
            return False

    # temporarily show an error message
    def show_error_message(self, message):
        current_message = self.message_lbl.cget("text") # get current message
        self.message_lbl.config(text=message, fg="red") # change message to error message
        self.root.after(2000, lambda: self.message_lbl.config(text=current_message, fg="black")) # change message back to original message after 2 seconds
        
    def start_solving(self):
        # Check if board is valid
        if not sudoku_algorithms.is_valid_board(self.get_board_values()):
            self.show_error_message("Invalid board. Please check your entries.")
            return
        # Destroy initial buttons and message label
        self.start_solving_btn.destroy()
        self.camera_btn.destroy()

        # Change message label
        self.message_lbl.config(text="Good luck!")

        # Make any non-empty entries read-only
        for i in range(9):
            for j in range(9):
                if self.entries[i][j].get() != "":
                    self.entries[i][j].config(state="readonly")

        # Store intial entries
        self.initial_entries = self.get_board_values()

        # Create solve, reset, and new puzzle buttons
        self.see_solution_btn = tk.Button(self.button_frame, text="See Solution", command=self.see_solution)
        self.see_solution_btn.grid(row=1, column=0, columnspan=9, padx=10, pady=10)
        self.reset_btn = tk.Button(self.button_frame, text="Reset", command=self.reset)
        self.reset_btn.grid(row=2, column=0, columnspan=9, padx=10, pady=10)
        self.new_puzzle_btn = tk.Button(self.button_frame, text="New Puzzle", command=self.new_puzzle)
        self.new_puzzle_btn.grid(row=3, column=0, columnspan=9, padx=10, pady=10)
        self.hint_btn = tk.Button(self.button_frame, text="Hint", command=self.show_hint)
        self.hint_btn.grid(row=4, column=0, columnspan=9, padx=10, pady=10)

    def see_solution(self):
        if sudoku_algorithms.is_valid_board(self.get_board_values()):
            current_board = self.get_board_values()
            sudoku_algorithms.solve(current_board)
            self.update_entries(current_board)
        else:
            self.show_error_message("Current board has no solution. Showing solution to initial board.")
            board = copy.deepcopy(self.initial_entries)
            sudoku_algorithms.solve(board)
            self.update_entries(board)
    
    def reset(self):
        self.update_entries(self.initial_entries)

    def show_hint(self):
        # check if board is valid
        if not sudoku_algorithms.is_valid_board(self.get_board_values()):
            self.show_error_message("Invalid board. Please check your entries.")
            return
        
        # get new board with hint
        new_board = sudoku_algorithms.random_hint(self.get_board_values())

        # check if there are any empty cells
        if new_board == None:
            self.show_error_message("No empty cells.")
            return
        
        self.update_entries(new_board)

    def new_puzzle(self):
        self.see_solution_btn.destroy()
        self.reset_btn.destroy()
        self.new_puzzle_btn.destroy()
        self.hint_btn.destroy()

        self.message_lbl.config(text="Enter known digits, then click 'Start Solving'")
        self.start_solving_btn = tk.Button(self.button_frame, text="Start Solving", command=self.start_solving)
        self.start_solving_btn.grid(row=1, column=0, columnspan=9, padx=10, pady=10)
        self.camera_btn = tk.Button(self.button_frame, text="Camera", command=self.open_camera)
        self.camera_btn.grid(row=2, column=0, columnspan=9, padx=10, pady=10)

        # Make all entries empty
        for i in range(9):
            for j in range(9):
                self.entries[i][j].config(state="normal")
                self.entries[i][j].delete(0, tk.END)
                
# display camera feed next to sudoku grid
    def open_camera(self):
        # change message label
        self.message_lbl.config(text="Take a picture of the sudoku board")
        # destroy buttons
        self.start_solving_btn.destroy()
        self.camera_btn.destroy()

        # create frame for camera feed
        self.camera_frame = tk.Frame(self.button_frame)
        self.camera_frame.grid(row=1, column=0, columnspan=9, padx=10, pady=10)

        # create frame for buttons
        self.button_frame2 = tk.Frame(self.button_frame)
        self.button_frame2.grid(row=2, column=0, columnspan=9, padx=10, pady=10)

        # create button to take picture
        self.take_picture_btn = tk.Button(self.button_frame2, text="Take Picture", command=self.take_picture)
        self.take_picture_btn.grid(row=0, column=0, padx=10, pady=10)
        self.back_btn = tk.Button(self.button_frame2, text="Back", command=self.back)
        self.back_btn.grid(row=0, column=1, padx=10, pady=10)

        # create camera feed
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 252)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 252)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        # create canvas to display camera feed
        self.camera_canvas = tk.Canvas(self.camera_frame, width=252, height=252)
        self.camera_canvas.grid(row=0, column=0, padx=10, pady=10)

        # variable to keep track of whether camera is running
        self.camera_running = True

        # create timer to update camera feed
        self.update_camera()
    
    def update_camera(self):
        if self.camera_running == False:
            return
        
        # get frame from camera
        _, frame = self.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        # update canvas with new frame
        self.camera_canvas.create_image(0, 0, image=frame, anchor="nw")
        self.camera_canvas.image = frame

        # update camera feed every 15 milliseconds
        self.root.after(15, self.update_camera)
    
    def take_picture(self):
        # take and display picture
        self.camera_running = False
        _, frame = self.camera.read()
        # crop image to 252x252
        frame = frame[0:252, 0:252]
        predicted = image_processing.predict_all(frame)
        self.update_entries(predicted)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        self.camera_canvas.create_image(0, 0, image=frame, anchor="nw")
        self.camera_canvas.image = frame


    def back(self):
        # stop camera feed
        self.camera_running = False
        self.camera.release()

        # destroy camera feed and buttons
        self.camera_frame.destroy()
        self.button_frame2.destroy()

        # change message label
        self.message_lbl.config(text="Enter known digits, then click 'Start Solving'")

        # create initial buttons
        self.start_solving_btn = tk.Button(self.button_frame, text="Start Solving", command=self.start_solving)
        self.start_solving_btn.grid(row=1, column=0, columnspan=9, padx=10, pady=10)
        self.camera_btn = tk.Button(self.button_frame, text="Camera", command=self.open_camera)
        self.camera_btn.grid(row=2, column=0, columnspan=9, padx=10, pady=10)


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
                if board[i][j] == 0:
                    self.entries[i][j].insert(0, "") # insert empty string
                else:
                    self.entries[i][j].insert(0, board[i][j]) # insert new value
    