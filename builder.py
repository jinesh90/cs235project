"""
class that build model.train model, eval and save as bin for future use.
"""
import torch.nn as nn
import torch
import numpy as np
from collections import defaultdict
from config.parameters import EPOCHS, DEVICE, SAVE_PATH
import matplotlib.pyplot as plt


class ModelBuilder:
    """
    model training main class
    """
    def __init__(self, model=None, loss_function=None, optimizer=None,
                 model_scheduler=None, model_save=True, model_name="model.bin"):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = model_scheduler
        self.training_accuracy = 0
        self.training_loss = 0
        self.validation_accuracy = 0
        self.validation_loss = 0
        self.model_save = model_save
        self.model_name = model_name
        self.model_training_data = defaultdict(list)

    def train(self, loader, data_len):
        """
        model training function
        :param loader:
        :param data_len:
        :return:
        """
        self.model = self.model.train()
        train_loss = []
        true_pred = 0
        for data in loader:
            ip = data["input_ids"].to(DEVICE)
            mask = data["attention_mask"].to(DEVICE)
            sentiment = data["sentiment"].to(DEVICE)
            predicted_output = self.model(input_ids=ip, mask=mask)
            op, prediction = torch.max(predicted_output, dim=1)
            train_loss_for_data = self.loss_function(predicted_output, sentiment)
            true_pred += torch.sum(torch.tensor(prediction == sentiment))
            train_loss.append(train_loss_for_data.item())
            train_loss_for_data.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return true_pred.double() / data_len, np.mean(train_loss)

    def eval(self, loader, data_len):
        """
        model evaluator.
        :param loader:
        :param data_len:
        :return:
        """
        self.model = self.model.eval()
        eval_loss = []
        true_pred = 0
        with torch.no_grad():
            for data in loader:
                ip = data.get("input_ids").to(DEVICE)
                mask = data.get("attention_mask").to(DEVICE)
                sentiment = data.get("sentiment").to(DEVICE)
                predicted_output = self.model(input_ids=ip, mask=mask)
                op, prediction = torch.max(predicted_output, dim=1)
                eval_loss_for_data = self.loss_function(predicted_output, sentiment)
                true_pred += torch.sum(torch.tensor(prediction == sentiment))
                eval_loss.append(eval_loss_for_data.item())
        return true_pred.double() / data_len, np.mean(eval_loss)

    def start(self, train_loader_obj, val_loader_obj, train_data_len, val_data_len):
        """
        start training and eval here
        :return:
        """
        base_accuracy = 0
        for i in range(EPOCHS):
            print('-' * 100)
            print("Epoch number: {}/{}".format(i+1, EPOCHS))
            print('-' * 100)
            self.training_accuracy, self.training_loss = self.train(train_loader_obj, train_data_len)

            print("Training loss: {} Training Accuracy: {}".format(self.training_loss, self.training_accuracy))
            self.validation_accuracy, self.validation_loss = self.eval(val_loader_obj, val_data_len)

            print("Validation loss: {}, Validation Accuracy: {}".format(self.validation_loss, self.validation_accuracy))
            #print(f'Validation loss {self.validation_loss} accuracy {self.validation_accuracy}')
            self.model_training_data['train_accuracy'].append(self.training_accuracy)
            self.model_training_data['train_loss'].append(self.training_loss)
            self.model_training_data['validation_accuracy'].append(self.validation_accuracy)
            self.model_training_data['validation_loss'].append(self.validation_loss)

        plt.plot(self.model_training_data['train_accuracy'], label="train accuracy")
        plt.plot(self.model_training_data['validation_accuracy'], label="validation accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(color='red', linestyle='-.', linewidth=0.5)
        plt.ylim([0, 1])
        plt.savefig("bin/{}_accuracy.png".format(self.model_name))

        plt.plot(self.model_training_data['train_loss'], label="training loss")
        plt.plot(self.model_training_data['validation_loss'], label="validation loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid(color='red', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.savefig("bin/{}_loss.png".format(self.model_name))

        if self.validation_accuracy > base_accuracy:
            if self.model_save:
                torch.save(self.model.state_dict(), SAVE_PATH + self.model_name)
                base_accuracy = self.validation_accuracy