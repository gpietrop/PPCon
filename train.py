import os
import math
from IPython import display

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adadelta
from torch.nn.functional import mse_loss

from conv1med import Conv1dMed
from mlp import MLPDay, MLPYear, MLPLat, MLPLon


path = "result"  # result directory
if not os.path.exists(path):
    os.mkdir(path)

train_dataset = [[torch.rand([1, 3, 200]), torch.rand([1, 1, 200])]]  # shape of the signal
test_dataset = [torch.rand([1, 3, 200]), torch.rand([1, 1, 200])]  # shape of the signal

training_day = torch.rand([1])
training_year = torch.rand([1])
training_lat = torch.rand([1])
training_lon = torch.rand([1])

testing_day = torch.rand([1])
testing_year = torch.rand([1])
testing_lat = torch.rand([1])
testing_lon = torch.rand([1])
# test_dataset =

alpha = torch.tensor(4e-4)
lr = 0.1
epoch = 100
snaperiod = 20

model_mlp_day = MLPDay()
model_mlp_year = MLPYear()
model_mlp_lat = MLPLat()
model_mlp_lon = MLPLon()

model_conv = Conv1dMed()

params = list(model_mlp_day.parameters()) + list(model_mlp_year.parameters()) + list(model_mlp_lat.parameters()) + list(
    model_mlp_lon.parameters()) + list(model_conv.parameters())
optimizer = Adadelta(params=params, lr=lr)

loss_train = []
loss_test = []

f, f_test = open(path + "/train_loss.txt", "w+"), open(path + "/test_loss.txt", "w+")

for ep in range(epoch):
    for training_x, training_y in train_dataset:

        output_day = model_mlp_day(training_day.float())
        output_year = model_mlp_year(training_year.float())
        output_lat = model_mlp_lat(training_lat.float())
        output_lon = model_mlp_lon(training_lon.float())

        output_day = output_day.unsqueeze(0).unsqueeze(0)
        output_year = output_year.unsqueeze(0).unsqueeze(0)
        output_lat = output_lat.unsqueeze(0).unsqueeze(0)
        output_lon = output_lon.unsqueeze(0).unsqueeze(0)

        training_x = torch.cat((training_x, output_day, output_year, output_lat, output_lon), 1)
        output = model_conv(training_x.float())

        loss_conv = mse_loss(training_y, output)  # MSE
        loss_train.append(loss_conv.item())

        # print(model_conv.conv3.weight.mean())
        # print(model_mlp_lat.network[0].weight.mean())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_conv.item():.12f}")
        display.clear_output(wait=True)
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_conv.item():.12f} \n")

        optimizer.zero_grad()
        loss_conv.backward()
        optimizer.step()

        # print(model_conv.conv1.weight[0])
    # test
    if ep % snaperiod == 0:
        model_mlp_day.eval()
        model_mlp_year.eval()
        model_mlp_lat.eval()
        model_mlp_lon.eval()

        model_conv.eval()

        with torch.no_grad():
            testing_x = test_dataset[0]
            testing_y = test_dataset[1]

            output_day_test = model_mlp_day(testing_day.float())
            output_year_test = model_mlp_year(testing_year.float())
            output_lat_test = model_mlp_lat(testing_lat.float())
            output_lon_test = model_mlp_lon(testing_lon.float())

            output_day_test = output_day_test.unsqueeze(0).unsqueeze(0)
            output_year_test = output_year_test.unsqueeze(0).unsqueeze(0)
            output_lat_test = output_lat_test.unsqueeze(0).unsqueeze(0)
            output_lon_test = output_lon_test.unsqueeze(0).unsqueeze(0)

            testing_x = torch.cat((testing_x, output_day_test, output_year_test, output_lat_test, output_lon_test), 1)

            output_test = model_conv(testing_x.float())

            loss_conv = mse_loss(testing_y, output_test)
            loss_test.append(loss_conv)

            print(f"-----[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_conv.item():.12f}")
            display.clear_output(wait=True)
            f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_conv.item():.12f} \n")

f.close()
f_test.close()

# torch.save(model.state_dict(), "model/model2015/model_step1_ep_" + str(epoch_c + epoch_pretrain) + ".pt")
