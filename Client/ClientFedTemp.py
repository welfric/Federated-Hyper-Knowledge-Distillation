import torch
import copy
import torch.nn as nn
from utils import Accuracy
from models import TempNet


class ClientFedTemp:
    def __init__(self, args, model, train_loader, test_loader, idx, device):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.idx = idx
        self.device = device
        self.model = copy.deepcopy(model)
        self.tempnet = TempNet(feature_dim=64).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.warmup_rounds = 25

    def update_weights(self, global_round):
        self.model.train()

        warmup = global_round < self.warmup_rounds

        if warmup:
            self.tempnet.train()
        else:
            self.tempnet.eval()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.temp_optimizer = torch.optim.SGD(
            self.tempnet.parameters(),
            lr=self.args.lr/10,
        )

        epoch_loss = []
        for _ in range(self.args.local_ep):
            batch_loss = []
            for X, y in self.train_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                if warmup:
                    self.temp_optimizer.zero_grad()

                # Forward pass through backbone
                features, logits = self.model(X)

                # Get τ from TempNet
                tau = self.tempnet(features.detach())

                # Scale logits by τ
                scaled_logits = logits / tau

                # Compute loss
                loss = self.criterion(scaled_logits, y)

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.args.clip_grad
                    )
                optimizer.step()
                if warmup:
                    self.temp_optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        with torch.no_grad():

            sample_data, _ = next(iter(self.train_loader))
            sample_data = sample_data.to(self.device)
            f, _ = self.model(sample_data)
            tau_val = self.tempnet(f).item()

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), tau_val

    def test_accuracy(self):
        self.model.eval()
        total_acc = 0
        total_batches = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                logits = (
                    outputs[1]
                    if isinstance(outputs, (list, tuple)) and len(outputs) > 1
                    else outputs
                )
                preds = logits.argmax(dim=1)
                total_acc += Accuracy(y, preds)
                total_batches += 1
        return total_acc / total_batches if total_batches > 0 else 0

    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)
