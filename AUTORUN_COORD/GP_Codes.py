import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_task):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), #LinearMean(input_size=3)
            num_tasks=num_task,
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            num_tasks=num_task,
            rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':

    num_task = 24
    type = "BM"
    array = np.loadtxt(type+"_train_codes_AFTER.txt", dtype="float32")
    train_y = torch.from_numpy(array[:, 3:3+24])
    train_x = torch.from_numpy(array[:, 3+24:])
    array = np.loadtxt(type+"_train_codes_PRE_ADD.txt", dtype="float32")
    eval_y = torch.from_numpy(array[:, 3:3+24])
    eval_x = torch.from_numpy(array[:, 3+24:])

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_task)
    model = MultitaskGPModel(train_x, train_y, likelihood, num_task)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 15000
    # label_y_numpy = train_y.numpy().reshape(shape)

    now = time.perf_counter()
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if i%100 == 0:
            print('Iter %d/%d - Loss: %.3f' % (
                i + 1, training_iter, loss.item(),
            ))
        optimizer.step()

    print("training time", time.perf_counter() - now)

    model.eval()
    likelihood.eval()
    input = train_x
    predicted = model(input)
    mean = predicted.mean.detach().cpu().numpy()
    low, up = predicted.confidence_region()
    # eval_shape = (100, 100, 2)
    np.savetxt(type+"_train_codes_AFTER_mean.txt", mean)
    np.savetxt(type+"_train_codes_AFTER_low.txt", low.detach().cpu().numpy())
    np.savetxt(type+"_train_codes_AFTER_up.txt", up.detach().cpu().numpy())


    input = eval_x
    predicted = model(input)
    mean = predicted.mean.detach().cpu().numpy()
    low, up = predicted.confidence_region()
    # eval_shape = (100, 100, 2)
    np.savetxt(type+"_train_codes_AFTER_ADD_mean.txt", mean)
    np.savetxt(type+"_train_codes_AFTER_ADD_low.txt", low.detach().cpu().numpy())
    np.savetxt(type+"_train_codes_AFTER_ADD_up.txt", up.detach().cpu().numpy())

    num_task = 24
    type = "NB"
    array = np.loadtxt(type + "_train_codes_AFTER.txt", dtype="float32")
    train_y = torch.from_numpy(array[:, 3:3+24])
    train_x = torch.from_numpy(array[:, 3+24:])
    array = np.loadtxt(type+"_train_codes_PRE_ADD.txt", dtype="float32")
    eval_y = torch.from_numpy(array[:, 3:3+24])
    eval_x = torch.from_numpy(array[:, 3+24:])

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_task)
    model = MultitaskGPModel(train_x, train_y, likelihood, num_task)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 15000
    # label_y_numpy = train_y.numpy().reshape(shape)

    now = time.perf_counter()
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if i%100 == 0:
            print('Iter %d/%d - Loss: %.3f' % (
                i + 1, training_iter, loss.item(),
            ))
        optimizer.step()

    print("training time", time.perf_counter() - now)

    model.eval()
    likelihood.eval()
    input = train_x
    predicted = model(input)
    mean = predicted.mean.detach().cpu().numpy()
    low, up = predicted.confidence_region()
    # eval_shape = (100, 100, 2)
    np.savetxt(type + "_train_codes_AFTER_mean.txt", mean)
    np.savetxt(type + "_train_codes_AFTER_low.txt", low.detach().cpu().numpy())
    np.savetxt(type + "_train_codes_AFTER_up.txt", up.detach().cpu().numpy())

    input = eval_x
    predicted = model(input)
    mean = predicted.mean.detach().cpu().numpy()
    low, up = predicted.confidence_region()
    # eval_shape = (100, 100, 2)
    np.savetxt(type + "_train_codes_AFTER_ADD_mean.txt", mean)
    np.savetxt(type + "_train_codes_AFTER_ADD_low.txt", low.detach().cpu().numpy())
    np.savetxt(type + "_train_codes_AFTER_ADD_up.txt", up.detach().cpu().numpy())


    num_task = 24
    type = "PR"
    array = np.loadtxt(type + "_train_codes_AFTER.txt", dtype="float32")
    train_y = torch.from_numpy(array[:, 3:3+24])
    train_x = torch.from_numpy(array[:, 3+24:])
    array = np.loadtxt(type+"_train_codes_PRE_ADD.txt", dtype="float32")
    eval_y = torch.from_numpy(array[:, 3:3+24])
    eval_x = torch.from_numpy(array[:, 3+24:])

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_task)
    model = MultitaskGPModel(train_x, train_y, likelihood, num_task)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 15000
    # label_y_numpy = train_y.numpy().reshape(shape)

    now = time.perf_counter()
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if i%100 == 0:
            print('Iter %d/%d - Loss: %.3f' % (
                i + 1, training_iter, loss.item(),
            ))
        optimizer.step()

    print("training time", time.perf_counter() - now)

    model.eval()
    likelihood.eval()
    input = train_x
    predicted = model(input)
    mean = predicted.mean.detach().cpu().numpy()
    low, up = predicted.confidence_region()
    # eval_shape = (100, 100, 2)
    np.savetxt(type + "_train_codes_AFTER_mean.txt", mean)
    np.savetxt(type + "_train_codes_AFTER_low.txt", low.detach().cpu().numpy())
    np.savetxt(type + "_train_codes_AFTER_up.txt", up.detach().cpu().numpy())

    input = eval_x
    predicted = model(input)
    mean = predicted.mean.detach().cpu().numpy()
    low, up = predicted.confidence_region()
    # eval_shape = (100, 100, 2)
    np.savetxt(type + "_train_codes_AFTER_ADD_mean.txt", mean)
    np.savetxt(type + "_train_codes_AFTER_ADD_low.txt", low.detach().cpu().numpy())
    np.savetxt(type + "_train_codes_AFTER_ADD_up.txt", up.detach().cpu().numpy())
