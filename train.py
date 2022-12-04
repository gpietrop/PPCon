import os
import math
from IPython import display

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adadelta
from torch.nn.functional import mse_loss


from conv1med import Conv1dMed
# from normalization import Normalization


path = "result"  # result directory
if not os.path.exists(path):
    os.mkdir(path)

train_dataset = [[torch.rand([1, 7, 200]), torch.rand([1, 1, 200])]]  # shape of the signal
test_dataset = [torch.rand([1, 7, 200]), torch.rand([1, 1, 200])]  # shape of the signal
# test_dataset =

alpha = torch.tensor(4e-4)
lr = 0.001
epoch = 100
snaperiod = 20

model = Conv1dMed()
optimizer = Adadelta(model.parameters(), lr=lr)

loss_train = []
loss_test = []


f, f_test = open(path + "/train_loss.txt", "w+"), open(path + "/test_loss.txt", "w+")

for ep in range(epoch):
    for training_x, training_y in train_dataset:

        output = model(training_x.float())

        loss = mse_loss(training_y, output)  # MSE

        loss_train.append(loss.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss.item():.12f}")
        display.clear_output(wait=True)
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss.item():.12f} \n")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test
    if ep % snaperiod == 0:
        model.eval()
        with torch.no_grad():
            testing_x = test_dataset[0]
            testing_y = test_dataset[1]
            output_test = model(testing_x.float())

            loss = mse_loss(testing_y, output_test)

            loss_test.append(loss)

            print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss.item():.12f}")
            display.clear_output(wait=True)
            f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss.item():.12f} \n")

f.close()
f_test.close()

# torch.save(model.state_dict(), "model/model2015/model_step1_ep_" + str(epoch_c + epoch_pretrain) + ".pt")

