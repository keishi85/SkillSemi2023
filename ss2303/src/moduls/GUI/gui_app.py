import cv2
import tkinter as tk
import tkinter.filedialog
from tkinter import simpledialog, ttk
from PIL import Image, ImageTk
from ..controllers.detect_face_img import ImageProcess

class GuiApplication(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)

        self.master.title("Face image detection application")   # Window title
        self.master.geometry("516x550")           # window size
        self.create_widgets()
        self.input_img_path = None

        # An instance of ImageProcess
        self.face_detection = None
        self.drawn_img = None
        self.blured_img = None
        self.swapped_img = None

    # create widgets
    def create_widgets(self):
        # cancvas
        self.canvas = tk.Canvas(self.master,
                                bg="#000",
                                width=512,
                                height=512)
        self.canvas.place(x=0, y=0)

        # label
        self.label1 = tk.Label(text='File name:')
        self.label1.place(x=5, y=480, height=28)
        
        # textboRx
        self.text_box1 = tk.Entry(width=39)
        self.text_box1.place(x=70, y=480)

        # button
        self.btn1 = tk.Button(text="Select", command = self.select_file, width=4)
        self.btn1.place(x=377, y=480)
        self.btn2 = tk.Button(text="Show", command = self.open_imgge_file, width=4, state=tk.DISABLED)
        self.btn2.place(x=445, y=480)

        # Reset button
        self.btn_reset = tk.Button(text="Reset", command=self.reset_img, width=4, state=tk.DISABLED)
        self.btn_reset.place(x=241, y=515)

        # Detect button
        self.btn_detect = tk.Button(text="Detect", command=self.detect_face, width=4, state=tk.DISABLED)
        self.btn_detect.place(x=309, y=515)

        # Process button
        self.btn_process = tk.Button(text='Process', command=self.show_processing_options, width=4, state=tk.DISABLED)
        self.btn_process.place(x=377, y=515)

        # Save button
        self.btn_save = tk.Button(text='Save', command=self.save_img, width=4, state=tk.DISABLED)
        self.btn_save.place(x=445, y=515)

        # Create slider
        self.blur_label = tk.Label(self.master, text='Blur Strength')
        self.blur_label.place(x=40, y=518)
        self.blur_label.place_forget()

        self.blur_slider = tk.Scale(self.master, from_=0, to_=50, orient=tk.HORIZONTAL, showvalue=False)
        self.blur_slider.bind("<ButtonRelease-1>", self.update_blur_strength)
        self.blur_slider.place(x=135, y=522)
        self.blur_slider.place_forget()


    def select_file(self):
        file_name = tk.filedialog.askopenfilename()
        self.input_img_path = file_name
        if file_name:
            parts = file_name.split('/')
            if len(parts) > 3:
                file_name = '.../' + '/'.join(parts[-3:])
        self.text_box1.delete(0, tk.END)
        self.text_box1.insert(0, file_name)

        # Create instance
        self.face_detection = ImageProcess(self.input_img_path)

        # Change the button
        self.btn2.config(state=tk.NORMAL)

    def open_imgge_file(self):
        # Load image data using PIL.Image
        file_name = self.input_img_path
        cv2_image = cv2.imread(file_name)

        # Multiply window size/image size when outputting
        height, width = cv2_image.shape[:2]
        aspect_ratio = width / height
        new_height = 500
        new_width = int(new_height * aspect_ratio)
        cv2_image = cv2.resize(cv2_image, (new_width, new_height))
        
        # BGR(opencv) -> RGB(numpy) -> PIL image
        pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

        # convert PIL.Image to PhotoImage
        self.photo_image = ImageTk.PhotoImage(image=pil_image)

        # display image to canvas
        self.update() 
        self.canvas.create_image(self.canvas.winfo_width() // 2,
                                 self.canvas.winfo_height() // 2,
                                 image=self.photo_image)

        self.btn2.config(state=tk.DISABLED)
        self.btn_detect.config(state=tk.NORMAL)

    # Detect face
    def detect_face(self):
        self.face_detection.detect_face()
        self.face_detection.draw_img()

        # Get drawn image
        drawn_img = self.face_detection.get_drawn_img()

        # Multiply window size/image size when outputting
        height, width = drawn_img.shape[:2]
        aspect_ratio = width / height
        new_height = 500
        new_width = int(new_height * aspect_ratio)
        drawn_img = cv2.resize(drawn_img, (new_width, new_height))

        # convert PIL.Image to PhotoImage
        drawn_img_pil = Image.fromarray(cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB))
        self.drawn_img = ImageTk.PhotoImage(image=drawn_img_pil)

        # display image to canvas
        self.update()
        self.canvas.create_image(self.canvas.winfo_width() // 2,
                                 self.canvas.winfo_height() // 2,
                                 image=self.drawn_img)

        self.btn_reset.config(state=tk.NORMAL)
        self.btn_process.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)

    def show_processing_options(self):
        def on_ok():
            choice = option_var.get()
            if choice == 'Blur':
                self.blur_label.place(x=40, y=518)
                self.blur_slider.place(x=135, y=522)
                self.update_blur_strength()
            elif choice == 'Swap':
                self.swap_face()
            dialog.destroy()

        dialog = tk.Toplevel(self.master)
        dialog.title('Processing Options')
        dialog.geometry("280x130+200+420")

        option_var = tk.StringVar(dialog)
        option_var.set("Draw")     # Default value

        options = ["Blur", "Swap"]
        dropdown = ttk.Combobox(dialog, textvariable=option_var, values=options, state='readonly')
        dropdown.pack(side=tk.LEFT, padx=10)
        dropdown.set(options[0])

        ok_btn = tk.Button(dialog, text='OK', command=on_ok)
        ok_btn.pack(side=tk.LEFT)

        dialog.grab_set()

    def save_img(self):
        self.face_detection.save_data(kind='drawn')
        self.face_detection.save_data(kind='csv')

    # Reset image
    def reset_img(self):
        self.update()
        self.canvas.create_image(self.canvas.winfo_width() // 2,
                                 self.canvas.winfo_height() // 2,
                                 image=self.photo_image)

        self.btn_reset.config(state=tk.DISABLED)

        # Delete slider
        self.blur_label.place_forget()
        self.blur_slider.place_forget()

    # Blur face area
    def blur_face(self, strength):
        # Blur face
        self.face_detection.detect_face()
        self.face_detection.blur_face(strength)

        # Get blur image
        blur_img = self.face_detection.get_blur_img()

        # Multiply window size/image size when outputting
        height, width = blur_img.shape[:2]
        aspect_ratio = width / height
        new_height = 500
        new_width = int(new_height * aspect_ratio)
        blur_img = cv2.resize(blur_img, (new_width, new_height))

        # convert PIL.Image to PhotoImage
        blur_img_pil = Image.fromarray(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
        self.blured_img = ImageTk.PhotoImage(image=blur_img_pil)

        # display image to canvas
        self.update()
        self.canvas.create_image(self.canvas.winfo_width() // 2,
                                 self.canvas.winfo_height() // 2,
                                 image=self.blured_img)

        self.btn_reset.config(state=tk.NORMAL)

    def update_blur_strength(self, event=None):
        # When the slider is released, get the current blur strength and apply it
        strength = self.blur_slider.get()
        self.blur_face(strength)

    # Swap face
    def swap_face(self):
        filename = tk.filedialog.askopenfilename()
        self.face_detection.swap_face(filename)

        # Get swapped image
        swap_img = self.face_detection.get_swap_img()

        # Multiply window size/image size when outputting
        height, width = swap_img.shape[:2]
        aspect_ratio = width / height
        new_height = 500
        new_width = int(new_height * aspect_ratio)
        swap_img = cv2.resize(swap_img, (new_width, new_height))

        # convert PIL.Image to PhotoImage
        swap_img_pil = Image.fromarray(cv2.cvtColor(swap_img, cv2.COLOR_BGR2RGB))
        self.swapped_img = ImageTk.PhotoImage(image=swap_img_pil)

        # Display image to canvas
        self.update()
        self.canvas.create_image(self.canvas.winfo_width() //2,
                                 self.canvas.winfo_height() // 2,
                                 image=self.swapped_img)

        self.btn_reset.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = GuiApplication(master=root)
    app.mainloop()


if __name__ == "__main__":
    main()   