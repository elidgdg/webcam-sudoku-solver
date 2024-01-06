import gui
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sudoku Solver")

    game = gui.GUI(root)

    root.mainloop()