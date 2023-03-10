from tkinter import ttk
import tkinter as Tk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from adaline import Adaline

x = None
n = None
s = None
y = None

def draw_noise(scale):
    global x,n,s,y
    global ax, f
    
    x = np.linspace(0, scale*np.pi, num=5000)
    n = np.random.normal(scale=0.24, size=x.size)
    s = 1 * np.sin(x)
    y = 1 * np.sin(x) + n
    
    ax.plot(x, y, label='Total', color='#39ff14')
    ax.plot(x, s, label='Sine', color='#000000')
    f.canvas.draw()
    pass

def adaline_train(lr):
    global ax, f
    global y, x, s
    
    n_input = 10
    output = y[:n_input]
    adaline = Adaline(lr, 50)
    
    for i in range(0, x.size - n_input):
        inputs = np.column_stack([x[i:i+n_input], y[i:i+n_input]])
        desired = y[i:i+n_input]
        adaline.fitness(inputs, desired)
        aux = adaline.predict(inputs[-1])
        output = np.append(output, aux)
        
    ax.plot(x[:i+n_input+1], output, label="Adaline", color="#eb4034")
    ax.plot(x, s, label='Sine', color='#ffad1f')
    ax.legend()
    f.canvas.draw()
    
def ButtonTrain_Event():
    adaline_train(scaleLearningRate.get())
    
def ButtonDrawCurve_Event():
    global scaleScale, ax, f
    ax.clear()
    ax.grid('on')
    draw_noise(scaleScale.get())
    pass

def ScaleLearningRate_Event(event):
    print('Learning Rate: ', scaleLearningRate.get())

def ScaleScale_Event(event):
    print('Scale: ', scaleScale.get())

root = Tk.Tk()
root.geometry('1000x500')
root.minsize(1000,500)
root.title('Adaline')

f = Figure(figsize=(0,0), dpi=100)
ax = f.add_subplot(111)
ax.grid('on')

canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().place(relx=0.05, rely=0.05, relheight=0.90, relwidth=0.8)

scaleLearningRate = ttk.Scale(from_=0.003, to=0.005, command=ScaleLearningRate_Event)
scaleLearningRate.place(relx=0.875, rely=0.7, height=20, relwidth=0.1)

scaleScale = ttk.Scale(from_=0.5, to=2, command=ScaleScale_Event)
scaleScale.place(relx=0.875, rely=0.75, height=20, relwidth=0.1)

buttonDraw = ttk.Button(text='Desplegar', command=ButtonDrawCurve_Event)
buttonDraw.place(relx=0.875, rely=0.20, relheight=0.05, relwidth=0.1)

buttonTrain = ttk.Button(text='Entrenar', command=ButtonTrain_Event)
buttonTrain.place(relx=0.875, rely=0.25, relheight=0.05, relwidth=0.1)

draw_noise(2)
Tk.mainloop()