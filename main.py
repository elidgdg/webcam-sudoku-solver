import gui
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sudoku Solver")
    root.geometry("435x291")
    root.resizable(False, False)

    game = gui.GUI(root)

    root.mainloop()