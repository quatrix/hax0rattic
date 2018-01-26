import tkinter as tk
from PIL import ImageTk, Image
import os



class MainWindow():
    def all_images(self):
        d = './chars/'

        for i in filter(lambda x: x.endswith('.png'), os.listdir(d)):
            self.current_file = os.path.join(d, i)
            img = Image.open(self.current_file)
            img = img.resize((250, 250), Image.ANTIALIAS)
            yield ImageTk.PhotoImage(img)

    def __init__(self, main):
        # canvas for image
        self.canvas = tk.Canvas(main, width=400, height=400)
        self.canvas.grid(row=0, column=0)


        # set first image on canvas
        self.images = self.all_images()
        self.current_image = next(self.images)

        self.image_on_canvas = self.canvas.create_image(0, 0, anchor = tk.NW, image = self.current_image)


    def classify_image(self, e):
        # change image

        possible_chars = [
            '0', '1', '2', '3', 
            '4', '5', '6', '7', 
            '8', '9', '+', '-',
            '*', '/'
        ]

        pressed_char = e.char

        if pressed_char not in possible_chars:
            print('nay')
            return

        if pressed_char == '+':
            pressed_char = 'plus'
        elif pressed_char == '-':
            pressed_char = 'minus'
        elif pressed_char == '*':
            pressed_char = 'mul'
        elif pressed_char == '/':
            pressed_char = 'div'


        new_fname = './chars/{}/{}'.format(pressed_char, os.path.basename(self.current_file))
        print('moving {} to {}'.format(self.current_file, new_fname))
        os.rename(self.current_file, new_fname)
        self.current_image = next(self.images)
        self.canvas.itemconfig(self.image_on_canvas, image = self.current_image)

root = tk.Tk()
m = MainWindow(root)

root.bind("<KeyPress>", m.classify_image)
root.mainloop()
