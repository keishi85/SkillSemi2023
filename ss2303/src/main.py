import tkinter as tk
from moduls.GUI.gui_app import GuiApplication


def main():
    root = tk.Tk()
    app = GuiApplication(master=root)
    app.mainloop()


if __name__ == '__main__':
    main()