import csv;
import math
import pandas as pd
import numpy as np

class NaiveBayes:


   #initialize the user inputs
    def __init__(self,path,numOfBins):
        self.directoryName = path
        self.numBins=numOfBins

    def loadCsv(self):
        dataSet = pd.read_csv(self.directoryName+"\\train.csv")
        return dataSet

    #recognize the variables categories
    def categorizeAttributes(self):
        self.features = []
        self.classPredicte=""
        structure = open(self.directoryName+"\\Structure.txt","r")
        currLine = structure.readline()
        while currLine:
            line = currLine.split()
            self.features.append(line[1])
            currLine = structure.readline()
        self.classPredicte = self.features[len(self.features)-1]
        del self.features[len(self.features)-1]

    #find mean for numric categories
    def stdev(self,numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    #complete missing values
    def missingValues(self,dataSet):
        for i in dataSet:
            if((dataSet[i].dtype != np.float64) and (dataSet[i].dtype != np.int64)):
                mostFreq = dataSet[i].mode()[0]
                dataSet[i].fillna(mostFreq,inplace=True)
            else:
                colMean = dataSet[i].mean()
                colMean = math.floor(colMean)
                dataSet[i].fillna(colMean,inplace=True)
                self.partition(i, dataSet ,self.numBins)

    #discretization
    def binning(self, col, cut_points):
        minVal = col.min()
        maxVal = col.max()
        break_points_list = list()
        break_points_list.append(minVal)
        for i in cut_points:
            break_points_list.append(i)
        break_points_list.append(maxVal)
        break_points = np.asarray(break_points_list)
        labels = range(len(cut_points)+1)
        colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
        return colBin

    #partiotion for equal width
    def partition(self, col, dataset, numberOfBins):
        max = (dataset[col]).max()
        min = (dataset[col]).min()
        binWidth = (max-min)/numberOfBins
        bins = list()
        i = 0
        i = i+1
        while i < numberOfBins-1:
            min = min + binWidth
            bins.append(min)
            i = i + 1
        bins_arr = np.asarray(bins)
        dataset[col] = self.binning(dataset[col], bins_arr)


    #classifaier the model according to naive bayes
    def classifaier(self):
        self.df = self.loadCsv()
        self.missingValues(self.df)
        self.tst = pd.read_csv(self.directoryName+"\\test.csv")
        self.missingValues(self.tst)
        del self.tst['class']
        num_of_att = 0
        name_of_att = list()
        for col in (self.tst).columns:
            name_of_att.append(col)
            num_of_att = num_of_att+1
        prob_for_yes = float(len(self.df[self.df['class'] == 'Y']))/len(self.df['class'])
        prob_for_no = float(len(self.df[self.df['class'] == 'N']))/len(self.df['class'])
        m = 2
        final_test_predict=list()
        df_matrix=(self.tst).as_matrix()
        for i in df_matrix:
            result_for_yes = list()
            result_for_no = list()
            count_column=0
            for j in i:
              Nc_true_arr = (self.df).loc[(self.df[name_of_att[count_column]] == j) & (self.df['class'] == 'Y')]
              Nc_true = len(Nc_true_arr)
              Nc_false_arr = (self.df).loc[(self.df[name_of_att[count_column]] == j) & (self.df['class'] == 'N')]
              Nc_false = len(Nc_false_arr)
              p= float(1)/self.df[name_of_att[count_column]].nunique()
              N_true = len(self.df[self.df['class'] == 'Y'])
              N_false = len(self.df[self.df['class'] == 'N'])
              prob_cond_conditional_true = (Nc_true + (m * p))/(N_true+m)
              result_for_yes.append(prob_cond_conditional_true)
              prob_cond_conditional_false = (Nc_false + (m * p)) / (N_false + m)
              result_for_no.append(prob_cond_conditional_false)
              count_column+=1
            yes_predict = 1
            no_predict = 1
            for res_yes in result_for_yes:
                yes_predict = float(yes_predict) * res_yes
            yes_predict = float(yes_predict) * prob_for_yes
            for res_no in result_for_no:
                no_predict = float(no_predict) * res_no
            no_predict = float(no_predict) * prob_for_no
            if(yes_predict > no_predict):
                prediction = 'yes'
            else:
                prediction = 'no'
            final_test_predict.append(prediction)

        numOfRecord = 1
        text_file = open(self.directoryName+"\\output.txt", "w")
        for i in final_test_predict:
            text_file.write(str(numOfRecord)+" %s" % i+"\n")
            numOfRecord+=1
        text_file.close()
