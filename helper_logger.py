import os
import torch
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

class InternalDataPoint():
    def __init__(
            self, 
            epoch               : int, 
            training_loss       : float, 
            training_accuracy   : float, 
            validation_loss     : float, 
            validation_accuracy : float,
            time_stamp = datetime.now() 
        ) -> None:
    
        self.epoch               = epoch
        self.training_loss       = training_loss
        self.training_accuracy   = training_accuracy
        self.validation_loss     = validation_loss
        self.validation_accuracy = validation_accuracy
        self.time_stamp          = time_stamp

class DataLogger():
    def __init__(self, experiment_name : str, load_last = False) -> None:
        self.logs : list[InternalDataPoint] = []
        
        self.root_dir = self.__setup_dir__(experiment_name, load_last)

        self.current_best_accuracy = 0.0
        self.current_best_epoch    = 0
        self.current_epoch_is_best = False

        self.epoch_bias = 0

        if load_last:
            self.__load__()

        print("| Datalogger Setup Complete !")


    def __setup_dir__(self, experiment_name : str, force = False) -> str:
        base_dir = './runs'
        base_dir = os.path.abspath(base_dir)
        os.makedirs(base_dir, exist_ok = True)

        experiment_counter = 0
        for dir_entry in os.scandir(base_dir):
            if dir_entry.is_dir() and (experiment_name in dir_entry.name):
                experiment_counter += 1

        if force:
            experiment_run = f"{experiment_name}-{experiment_counter}"
            dpath = os.path.join(base_dir, experiment_run)
            dpath = os.path.abspath(dpath)
        else:
            experiment_run = f"{experiment_name}-{experiment_counter + 1}" 
            dpath = os.path.join(base_dir, experiment_run)
            dpath = os.path.abspath(dpath)
            os.makedirs(dpath)           

        return dpath

    def get_filepath(self, file_name : str) -> str:
        return os.path.join(self.root_dir, file_name)

    def latest_loss(self) -> float:
        if len(self.logs) > 0:
            return self.logs[-1].training_loss
        return 0.0
        
    def best_accuracy(self) -> str:
        return f"{(self.current_best_accuracy * 100):.2f} %"

    def append(self, 
            epoch               : int, 
            training_loss       : float, 
            training_accuracy   : float, 
            validation_loss     : float, 
            validation_accuracy : float,
            time_stamp = datetime.now() 
        ) -> InternalDataPoint:

        self.current_epoch_is_best = False

        log = InternalDataPoint(
            epoch + self.epoch_bias,
            training_loss,
            training_accuracy,
            validation_loss,
            validation_accuracy,
            time_stamp
        )


        if validation_accuracy > self.current_best_accuracy:
            self.current_epoch_is_best = True
            self.current_best_epoch    = epoch
            self.current_best_accuracy = validation_accuracy

        self.logs.append(log)
        return log

    def __to_df__(self) -> pd.DataFrame:
        data = [x.__dict__ for x in self.logs]
        return pd.DataFrame(data)
    
    def __plot_loss__(self) -> None:
        training_loss   = [i.training_loss   for i in self.logs]
        validation_loss = [i.validation_loss for i in self.logs]
        epoch           = [i.epoch           for i in self.logs]
        
        fpath = os.path.join(self.root_dir, "loss.png")
        plt.plot(epoch, training_loss,   label = 'Training Loss')
        plt.plot(epoch, validation_loss, label = 'Validation Loss')
        plt.title('Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(fpath)
        plt.yscale('linear')
        plt.clf()

    def __plot_accuracy__(self) -> None:
        training_acc   = [i.training_accuracy   for i in self.logs]
        validation_acc = [i.validation_accuracy for i in self.logs]
        epoch          = [i.epoch               for i in self.logs]
        
        fpath = os.path.join(self.root_dir, "accuracy.png")
        plt.plot(epoch, training_acc,   label = 'Training Accuracy')
        plt.plot(epoch, validation_acc, label = 'Validation Accuracy')
        plt.plot(
            [self.current_best_epoch, self.current_best_epoch],
            plt.ylim(), 
            label = f'Best : {self.current_best_accuracy:.2f}'
        )
        plt.title('Accuracy')
        plt.yscale('log')
        plt.legend()
        plt.savefig(fpath)
        plt.yscale('linear')
        plt.clf()

    def save(self) -> None:
        self.__plot_loss__()
        self.__plot_accuracy__()

        dfx = self.__to_df__()
        dfx.to_csv( 
            os.path.join(self.root_dir, "log.csv"), 
            index = False
        )

    def __load__(self) -> None:
        df = pd.read_csv( os.path.join(self.root_dir, "log.csv") )
        for data in df.to_dict(orient='records'):
            time_stamp = datetime.strptime(data['time_stamp'], '%Y-%m-%d %H:%M:%S.%f')
            self.append(
                data['epoch'],
                data['training_loss'],
                data['training_accuracy'],
                data['validation_loss'],
                data['validation_accuracy'],
                time_stamp
            )
        self.epoch_bias = len(self.logs)
        print("| Loaded {} checkpoints".format(self.epoch_bias))
        print("| Best epoch: {}".format(self.current_best_epoch))

    def write_text(self, message : str) -> None:
        log_file = os.path.join(self.root_dir, "log.txt")
        with open(log_file, 'a+') as file:
            file.write(message)
            file.write("\n")
        print(message)

if __name__ == "__main__":
    print("Experiment Logger")

    log = DataLogger("BCE", False)

    import random
    import time

    for i in range(10):
        print(">", i)
        log.append(
            i,
            random.random(),
            random.random(),
            random.random(),
            random.random()
        )
        log.save()

        if log.current_epoch_is_best:
            print("> BEST !")
            log.write_text(f"best - {i}")
        time.sleep(1)
    
    import numpy as np

    x  = np.random.random((2, 2))
    xt = str(x)
    log.write_text(xt)
    
    print("Done !")    