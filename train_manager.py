import os,sys
from os.path import *
from glob import glob
from time import time
import matplotlib.pyplot as plt
import torch
import numpy as np

class TrainManager:
    def __init__(self,args,model,objects):
        print('CUDA:', torch.version.cuda)

        self.name = args.name
        os.makedirs(self.name, exist_ok=True)
        self.experiment_root = self.name
        print("Experiment root folder:{}".format(self.experiment_root))

        self.weight_save_dir = join(self.experiment_root, "weight")
        self.plot_save_dir = join(self.experiment_root, "plot")
        self.result_save_dir = join(self.experiment_root, "result")
        os.makedirs(self.weight_save_dir, exist_ok=True)
        os.makedirs(self.plot_save_dir, exist_ok=True)
        os.makedirs(self.result_save_dir, exist_ok=True)

        self.args = args
        self.model = model
        self.objects = objects

        self.epoch = args.epoch if hasattr(args, 'epoch') else 0
        self.current_epoch = 0
        self.start_epoch = 1

        self.logs = {}
        self.log_files = {}
        self.logs["train"] = os.path.join(self.experiment_root,"train.csv")
        self.logs["valid"] = os.path.join(self.experiment_root,"valid.csv")
        self.log_files["train"] = open(self.logs["train"],"a")
        self.log_files["valid"] = open(self.logs["valid"],"a")

    
    def write_log(self, name, line):
        self.log_files[name].write(line+"\n")
        self.log_files[name].flush()

    def write_csv_log(self, name, values):
        strs = [None]*len(values)
        for i, value in enumerate(values):
            if isinstance(value,float):
                strs[i] = "%06f" % value 
            else :
                strs[i] = str(value)
        self.write_log(name,",".join(strs))

    def plot_log(self, name):
        try:
            stride = 30
            data = np.loadtxt(self.logs[name],delimiter=",")
            data = data[:len(data)-(len(data)%stride)]
            x = list(range(0,len(data),stride))
            for i in range(data.shape[1]):
                data_ = data[:,i]
                data_ = np.reshape(data_, (-1,stride))
                data_ = np.average(data_, axis=1)
                plt.figure()
                plt.plot(x, data_)
                plt.ylim([0,max(data_)*1.2])
                plt.grid(True)
                plt.savefig(os.path.join(self.plot_save_dir, "%d.png" % i))
                plt.clf()
        except Exception:
            print("plot error")

    def plot_train_val(self):
        try:
            stride = 30
            y_train = np.loadtxt(self.logs["train"],delimiter=",")
            y_valid = np.loadtxt(self.logs["valid"],delimiter=",")

            y_train = y_train[:len(y_train)-(len(y_train)%stride)]
            x_train = list(range(0,len(y_train),stride))
            x_valid = y_valid[:,0]
            for i in range(y_train.shape[1]):
                y_train_ = y_train[:,i]
                y_train_ = np.reshape(y_train_, (-1,stride))
                y_train_ = np.average(y_train_, axis=1)
                plt.figure()
                plt.plot(x_train, y_train_, label="train")
                plt.plot(x_valid, y_valid[:,i+1],  label="valid")
                plt.ylim([0,max(y_train_)*1.2])
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(self.plot_save_dir, "%d.png" % i))
                plt.close()
        except Exception as e:
            print("plot error", e)

    def save_checkpoint(self, epoch, model, objects, args):
        file_name = "{}_{}epoch.pt".format(int(time()), epoch)
        file_name = os.path.join(self.weight_save_dir, file_name)
        output = {
            'epoch':epoch,
            'model':model.state_dict(),
            'options':args
        }
        for key in objects.keys():
            output[key] = objects[key].state_dict()
        torch.save(output, file_name)

    def load_checkpoint(self, file_name, model, objects):
        if file_name == "recent":
            pts = sorted(glob(os.path.join(self.weight_save_dir, "*.pt")))
            file_name = pts[-1]
            print("most recent weight will be loaded. %s" % file_name)
        checkpoint = torch.load(file_name)
        model.load_state_dict(checkpoint['model'],strict=False)
        for key in objects.keys():
            if objects[key] is not None and key in checkpoint:
                objects[key].load_state_dict(checkpoint[key])
        self.current_epoch = checkpoint['epoch']
        self.start_epoch = self.current_epoch + 1
        return checkpoint['epoch']

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_epoch >= self.start_epoch:
            if self.current_epoch % 1 ==0:
                self.save_checkpoint(self.current_epoch, self.model, self.objects, self.args)
            self.plot_train_val()

        self.current_epoch += 1
        print("epoch %d:" % self.current_epoch)
        if self.current_epoch > self.epoch:
            raise StopIteration

        return self.current_epoch
        
    def close(self):
        for e in self.log_files:
            e.close()
