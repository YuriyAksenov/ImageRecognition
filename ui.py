from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from NetworkLoader import save, load
from Network import Network
from Network import *
import MnistLoader
training_data, validation_data, test_data = MnistLoader.load_data_wrapper()
training_data = list(training_data)

network:Network = None;
output = None;

def startUI():
    def readFile():
        global network
        file_name = fd.askopenfilename()
        network = load(file_name)
        print(network)

    root = Tk()
    root.title("Neural Networks")
    root.geometry("800x600")

    def openImage():
        global network
        global output
        file_name = fd.askopenfilename()
        img=Image.open(file_name)
        img=img.resize((280,280), Image.ANTIALIAS)
        photoImage=ImageTk.PhotoImage(img)
        label = Label(root, image=photoImage)
        label.image=photoImage
        label.place(relx=0.5, rely=0.05)

        predictedNumber = predict(file_name, network)
        predictionOutputLabel['text'] = "Output: "+str(predictedNumber)
        output="Output: "+str(predictedNumber)

    # Starting network logic
    def train():
        global network
        netLayers=list(map(int,str(layers.get()).split(',')))
        print(netLayers)
        network = Network(netLayers)
        epochs = int(epochNumber.get())
        minibatch = int(minibatchSize.get())
        rate = float(learningRate.get())
        print(layers, epochs, minibatch,rate)
        network.run(training_data, epochs, minibatch, rate, test_data = test_data,monitor_evaluation_accuracy=True)
    
    def saveNet():
       global network
       save(network,"./epoch.json")
    #
    
    main_menu = Menu()

    file_menu = Menu(tearoff=0)
    file_menu.add_command(label="Save", command=saveNet)
    file_menu.add_command(label="Open file for prediction", command=openImage)
    file_menu.add_command(label="Import net config", command=readFile)

    main_menu.add_cascade(label="File", menu=file_menu)

    trainButton = Button(text="Train",
                         background="#555",
                         foreground="#ccc",
                         font="16",
                         width=27,
                         command=train
                         )
    trainButton.place(relx=0.02, rely=0.0)
   
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
    predictionOutputLabel = Label(text=output, justify="right")
    predictionOutputLabel.place(relx=.5, rely=.02)
    root.config(menu=main_menu)
    root.mainloop()

    
