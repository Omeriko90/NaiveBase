import os
import tkFileDialog
import tkMessageBox
from NaiveBayes import NaiveBayes
import Tkinter as tk


class Classifier:

    # checks that the directory has all the needed files
        def checkDirectory(self):
            return os.path.isfile(self.master.directory+'\\Structure.txt') and os.path.isfile(self.master.directory+'\\train.csv') and os.path.isfile(self.master.directory+'\\test.csv')
    # checks if the files are empty
        def emptyFile(self):
            return os.stat(self.master.directory + '\\Structure.txt').st_size==0 or os.stat(self.master.directory + '\\train.csv').st_size==0 or os.stat(self.master.directory + '\\test.csv').st_size==0

        # open a directory chooser and sets the path to the chosen directory
        def setPath(self):
            self.master.directory = tkFileDialog.askdirectory()
            self.pathTextBox.delete(0, len(self.pathTextBox.get()))
            self.pathTextBox.insert(0,self.master.directory)
            if not self.checkDirectory() or self.emptyFile():
                tkMessageBox.showinfo("Error","There is a problem with the files")
                self.pathTextBox.delete(0, len(self.pathTextBox.get()))

        # checks that the bins are legal number
        def checkValue(self,event):
            num = self.numOfBins.get()
            if "." in num or "/" in num or int(num) < 0:
                self.buildBtn.config(state='disabled')
                tkMessageBox.showinfo("Error", "Illegal value for bins")
            else:
                self.buildBtn.config(state='normal')

        # builds the model
        def buildModel(self):
            self.model = NaiveBayes(self.master.directory,int(self.numOfBins.get()))
            self.model.loadCsv()
            tkMessageBox.showinfo("Naive Bayes Classifier", "Building classifier using train-set is done!")

        # predicts the outcomes
        def classifiyModel(self):
            tkMessageBox.showinfo("Naive Bayes Classifier", "Classifying")
            self.model.classifaier()
            tkMessageBox.showinfo("Naive Bayes Classifier", "Classifying is done!")
            self.master.destroy()

        # init the gui
        def __init__(self,master):
            self.master = master
            self.model = ''
            master.title("Naive Bayes Classifier")
            self.pathlbl = tk.Label(master, text='Directory Path')
            self.pathTextBox = tk.Entry(master)
            self.browseBtn = tk.Button(master, text="Browse", command=self.setPath)
            self.discretizationlbl = tk.Label(master, text='Discretization Bins')
            bins = tk.StringVar()
            self.numOfBins = tk.Entry(master, textvariable=bins)
            self.numOfBins.bind('<FocusOut>', self.checkValue)
            self.buildBtn = tk.Button(master, text="Build", command=self.buildModel,state='disabled')
            self.classifyBtn = tk.Button(master, text="Classify", command=self.classifiyModel)
            self.pathlbl.grid(row=0,column=0,padx=10,pady=5)
            self.pathTextBox.grid(row=0,column=1,padx=10,pady=5)
            self.browseBtn.grid(row=0,column=2,padx=10,pady=5)
            self.discretizationlbl.grid(row=1,column=0,padx=10,pady=5)
            self.numOfBins.grid(row=1,column=1,padx=10,pady=5)
            self.buildBtn.grid(row=2,column=1,pady=5)
            self.classifyBtn.grid(row=3,column=1,pady=5)


root = tk.Tk()
gui = Classifier(root)
root.mainloop()
