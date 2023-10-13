import tkinter.filedialog as filedialog
import tkinter as tk
import subprocess

master = tk.Tk()


def RunMLP():
    wrapper = ['python', 'mlp.py']
    result1 = subprocess.Popen(wrapper,  stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out1, err1 = result1.communicate()
    status_wrapper=out1.decode("utf-8")
    tk.messagebox.showinfo("Execution terminée")


def RunCNN():
    wrapper = ['python', 'cnn.py']
    result1 = subprocess.Popen(wrapper,  stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out1, err1 = result1.communicate()
    status_wrapper=out1.decode("utf-8")
    tk.messagebox.showinfo("Execution terminée")





def input():
    input_path = tk.filedialog.askopenfilename()
    input_entry.delete(1, tk.END)  
    input_entry.insert(0, input_path) 

top_frame = tk.Frame(master)
bottom_frame = tk.Frame(master)
line = tk.Frame(master, height=1, width=400, bg="grey80", relief='groove')

input_path = tk.Label(top_frame, text="Ouvrir un fichier:")
input_entry = tk.Entry(top_frame, text="", width=40)
browse1 = tk.Button(top_frame, text="Ouvrir", command=input)


top_frame.pack(side=tk.TOP)
line.pack(pady=10)
bottom_frame.pack(side=tk.BOTTOM)


input_path.pack(pady=5)
input_entry.pack(pady=5)
browse1.pack(pady=5)



begin_button = tk.Button(bottom_frame, text='CNN',command=RunCNN)
begin_button.pack(padx=10,pady=10, fill=tk.X,side=tk.LEFT)



_button = tk.Button(bottom_frame, text='MLP',command=RunMLP)
_button.pack(padx=10,pady=10, fill=tk.X)

w = tk.Label(master, text="Pour prédire veuillez choisir un modèle:") 
w.pack()
w.pack(fill=tk.BOTH, expand=1, padx=100, pady=50)


master.title('Speech Emotion Recognition')
master.geometry("500x300")
master.mainloop()