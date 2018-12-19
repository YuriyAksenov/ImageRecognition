from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image


def startUI():
    def readFile():
        file_name = fd.askopenfilename()
        f = open(file_name)
        s = f.read()
        print(s)
        f.close()

    root = Tk()
    root.title("Neural Networks")
    root.geometry("800x600")

    def openImage():
        file_name = fd.askopenfilename()
        img=Image.open(file_name)
        img=img.resize((280,280), Image.ANTIALIAS)
        photoImage=ImageTk.PhotoImage(img)
        label= Label(root, image=photoImage)
        label.image=photoImage
        label.place(relx=0.5, rely=0.25)
    main_menu = Menu()

    file_menu = Menu(tearoff=0)
    file_menu.add_command(label="Save")
    file_menu.add_command(label="Open file for prediction", command=openImage)
    file_menu.add_command(label="Import net config", command=readFile)

    main_menu.add_cascade(label="File", menu=file_menu)

    trainButton = Button(text="Train",
                         background="#555",
                         foreground="#ccc",
                         font="16",
                         width=27
                         # command=train
                         )
    trainButton.place(relx=0.02, rely=0.0)

    predictButton = Button(text="Predict",
                           background="#555",
                           foreground="#ccc",
                           font="16",
                           width=27
                           # command=predict
                           )
    predictButton.place(relx=0.5, rely=0.0)

    # Layers input
    layers = StringVar()
    layersLabel = Label(text="Comma separated layers", justify="right")
    layersLabel.place(relx=.02, rely=.08)
    layersInput = Entry(textvariable=layers)
    layersInput.place(relx=.34, rely=.1, anchor="c")

    # Mini batch size
    minibatchSize = StringVar()
    minibatchSizeLabel = Label(text="Mini batch size", justify="right")
    minibatchSizeLabel.place(relx=.02, rely=.12)
    minibatchSizeInput = Entry(textvariable=minibatchSize)
    minibatchSizeInput.place(relx=.34, rely=.14, anchor="c")

    # Epoch number
    epochNumber = StringVar()
    epochNumberLabel = Label(text="Epoch number", justify="right")
    epochNumberLabel.place(relx=.02, rely=.16)
    epochNumberInput = Entry(textvariable=epochNumber)
    epochNumberInput.place(relx=.34, rely=.18, anchor="c")

    # Learning rate
    learningRate = StringVar()
    learningRateLabel = Label(text="Learning rate", justify="right")
    learningRateLabel.place(relx=.02, rely=.2)
    learningRateInput = Entry(textvariable=learningRate)
    learningRateInput.place(relx=.34, rely=.22, anchor="c")

    # Prediction output
    predictionOutputLabel = Label(text="Output:", justify="right")
    predictionOutputLabel.place(relx=.5, rely=.15)
    root.config(menu=main_menu)
    root.mainloop()
