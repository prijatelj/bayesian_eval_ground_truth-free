import math,csv,random
import sys
import os
import re

import numpy as np
import scipy as sp
#import read_distribution as cdis # Moved the file's contents into here.

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric', 'truth_inference', 'zheng_2017', 'CATD')
except KeyError:
    # TODO would need to implement this path being passed via args, but really it needs removed.
    HERE = None

class Conf_Aware:
    def __init__(self,e2wl,w2el,datatype):
        self.e2wl = e2wl
        self.w2el = w2el
        self.weight = dict()
        self.datatype = datatype

    def examples_truth_calculation(self):
        self.truth = dict()

        if self.datatype == 'continuous':
             for example, worker_label_set in self.e2wl.items():
                temp = 0
                for worker, label in worker_label_set:
                    temp = temp + self.weight[worker] * float(label)

                self.truth[example] = temp

        else:
            for example, worker_label_set in self.e2wl.items():
                temp = dict()
                for worker, label in worker_label_set:
                    if (label in temp):
                        temp[label] = temp[label] + self.weight[worker]
                    else:
                        temp[label] = self.weight[worker]

                max = 0
                for label, num in temp.items():
                    if num > max:
                        max = num

                candidate = []
                for label, num in temp.items():
                    if max == num:
                        candidate.append(label)

                self.truth[example] = random.choice(candidate)

    def workers_weight_calculation(self):

        weight_sum = 0
        for worker, example_label_set in self.w2el.items():
            ns = len(example_label_set)
            if ns <= 30:
                chi_s = self.chi_square_distribution[ns][1-self.alpha/2]
            else:
                chi_s = 0.5 * pow(self.normal_distribution[1-self.alpha/2] + pow(2*ns-1 , 0.5) ,2)
            #print ns, chi_s
            dif = 0
            for example, label in example_label_set:
                if self.datatype == 'continuous':
                    dif = dif + (self.truth[example]-float(label))**2
                else:
                    if self.truth[example]!=label:
                        dif = dif + 1
            # NOTE uncertain of point of this print out.
            #if dif==0:
            #    print(worker, ns, dif, chi_s / (dif + 0.00001))

            self.weight[worker] = chi_s / (dif + 0.000000001)
            weight_sum = weight_sum + self.weight[worker]

        for worker in self.w2el.keys():
            self.weight[worker] = self.weight[worker] / weight_sum


    def Init_truth(self):
        self.truth = dict()

        if self.datatype == 'continuous':
            for example, worker_label_set in self.e2wl.items():
                temp = []
                for _, label in worker_label_set:
                    temp.append(float(label))

                self.truth[example] = np.median(temp)  # using median as intial value
                #self.truth[example] = np.mean(temp)  # using mean as initial value


        else:
            # using majority voting to obtain initial value
            for example, worker_label_set in self.e2wl.items():
                temp = dict()
                for _, label in worker_label_set:
                    if (label in temp):
                        temp[label] = temp[label] + 1
                    else:
                        temp[label] = 1

                max = 0
                for label, num in temp.items():
                    if num > max:
                        max = num

                candidate = []
                for label, num in temp.items():
                    if max == num:
                        candidate.append(label)

                self.truth[example] = random.choice(candidate)


    def Run(self,alpha,iterr, random_seed):
        # Seed the random number generator for reproducible results.
        random.seed(random_seed)

        # TODO This needs replaced and sampled from the actual distributions.
        directory = HERE

        self.chi_square_conf, self.chi_square_distribution = read_chi_square_distribution(directory)
        self.normal_conf,self.normal_distribution = read_normal_distribution(directory)
        #self.chi_square_conf, self.chi_square_distribution = cdis.read_chi_square_distribution()
        #self.normal_conf,self.normal_distribution = cdis.read_normal_distribution()
        self.alpha = alpha

        self.Init_truth()
        while iterr > 0:
            #print getaccuracy(sys.argv[2], self.truth, datatype)

            self.workers_weight_calculation()
            self.examples_truth_calculation()

            iterr -= 1


        return self.truth, self.weight



###################################
# The above is the EM method (a class)
# The following are several external functions
###################################

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def read_chi_square_distribution(directory):
    #dir = os.path.split(sys.argv[0])[0]
    file = open(directory+'/chi-square distribution.txt','r')
    flag = 0
    chi_square_conf = []
    chi_square_dis = dict()
    for line in file.readlines():
        line=re.split('\t|\n',line)
        if (flag == 0):
            for i in range(13):
                chi_square_conf.append(float(line[i+1]))
            flag = 1
        else:
            free_degree=int(line[0])
            temp=dict()
            for i in range(13):
                if (isfloat(line[i+1])):
                    temp[chi_square_conf[i]]=float(line[i+1])
                else:
                    temp[chi_square_conf[i]]=0.000001
            chi_square_dis[free_degree]=temp
    file.close()
    return chi_square_conf, chi_square_dis

def read_normal_distribution(directory):
    #dir = os.path.split(sys.argv[0])[0]
    file = open(directory + '/normal distribution.txt','r')
    flag = 0
    normal_conf = []
    normal_dis  = dict()
    for line in file.readlines():
        line=re.split('\t|\n',line)
        if (flag == 0):
            for i in range(13):
                normal_conf.append(float(line[i]))
            flag = 1
        else:
            for i in range(13):
                normal_dis[normal_conf[i]]=float(line[i])


    file.close()
    return normal_conf, normal_dis

def getaccuracy(truthfile, predict_truth, datatype):
    e2truth = {}
    f = open(truthfile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    tcount = 0
    count = 0

    for e, ptruth in predict_truth.items():

        if e not in e2truth:
            continue

        count += 1

        if datatype=='continuous':
            tcount = tcount + (ptruth-float(e2truth[e]))**2 #root of mean squared error
            #tcount = tcount + math.fabs(ptruth-float(e2truth[e])) #mean of absolute error
        else:
            if ptruth == e2truth[e]:
                tcount += 1

    if datatype=='continuous':
        return pow(tcount/count,0.5)  #root of mean squared error
        #return tcount/count  #mean of absolute error
    else:
        return tcount*1.0/count

def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set=[]

    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, worker, label = line
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker,label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example,label])

        if label not in label_set:
            label_set.append(label)

    return e2wl,w2el,label_set


if __name__ == "__main__":

    # if len(sys.argv)>=4 and sys.argv[3] == 'continuous':
    #     datatype = r'continuous'
    # else:
    #     datatype = r'categorical'

    datafile = sys.argv[1]
    datatype = sys.argv[2]
    e2wl, w2el, label_set = gete2wlandw2el(datafile)
    e2lpd, w2q = Conf_Aware(e2wl,w2el,datatype).Run(0.05,100, 1234)

    print(w2q)
    print(e2lpd)

    # truthfile = sys.argv[2]
    # accuracy = getaccuracy(truthfile, predict_truth, datatype)
    # print accuracy
