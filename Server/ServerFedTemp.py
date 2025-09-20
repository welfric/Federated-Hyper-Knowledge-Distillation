import torch
import copy
import numpy as np
import time
from tqdm import tqdm
from Server.ServerBase import Server
from models import TempNet
from Client.ClientFedTemp import ClientFedTemp
from utils import average_weights
import matplotlib.pyplot as plt


class ServerFedTemp(Server):
    def __init__(
        self,
        args,
        global_model,
        loaders_train,
        loaders_local_test,
        loader_global_test,
        device,
    ):
        super().__init__(
            args,
            global_model,
            loaders_train,
            loaders_local_test,
            loader_global_test,
            device,
        )

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(
                ClientFedTemp(
                    self.args,
                    copy.deepcopy(self.global_model),
                    self.Loaders_train[idx],
                    self.Loaders_local_test[idx],
                    idx=idx,
                    device=self.device,
                )
            )

    def train(self):
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        client_temps = [[] for _ in range(self.args.num_clients)]
        for epoch in tqdm(range(self.args.num_epochs)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            print(f"\n | Global Training Round : {epoch+1} |\n")
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(
                range(self.args.num_clients), m, replace=False
            )
            for idx in idxs_users:
                if self.args.upload_model:
                    self.LocalModels[idx].load_model(global_weights)
                w, loss, tau_val = self.LocalModels[idx].update_weights(
                    global_round=epoch
                )
                client_temps[idx].append(tau_val)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # Update global weights
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print("average local test accuracy:", test_accuracy / self.args.num_clients)
            print("global test accuracy: ", self.global_test_accuracy())

        print("Training is completed.")
        end_time = time.time()
        print("running time: {} s ".format(end_time - start_time))

        plt.figure(figsize=(10, 6))
        for i in range(len(client_temps)):
            plt.plot(
                range(self.args.num_epochs),
                client_temps[i],
                marker="s",
                label=f"Client {i}",
            )
        plt.title("Client τ evolution")
        plt.legend()
        plt.grid(True)
        plt.savefig("Client_tau_evolution.png")
        plt.show()
