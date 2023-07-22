from torch.nn.modules.container import ModuleList
from torch.autograd.grad_mode import inference_mode
import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.3
bias = 0.9

X = torch.arange(0,3,0.02).unsqueeze(dim=1)
y = weight * X + bias
train_data= int(0.8 * len(X))
X_train,Y_train = X[:train_data],y[:train_data]
X_test,Y_test = X[train_data:],y[train_data:]
len(X_train),len(Y_train),len(X_test),len(Y_test)

def plot_predictions(train_X = X_train,
                     train_Y= Y_train,
                     test_X=X_test,
                     test_Y=Y_test,
                     predictions=None):

  plt.figure(figsize=(10,14))

  plt.scatter(train_X,train_Y,c="r",s=3,label="Training Data")
  plt.scatter(test_X,test_Y,c="b",s=3,label="Testing Data")
  if predictions is not None:
    plt.scatter(test_X,predictions,c="y",s=3,label="Predictions")

  plt.legend(prop={"size":14})


class linear_regression(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features=1,
                                  out_features=1)



  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.linear_layer(x)


model_X = linear_regression()


losfn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_X.parameters(),
                            lr=0.01)

torch.manual_seed(42)

epochs = 300
epoch_count =[]
train_loss_values =[]
test_loss_values =[]
for epoch in range(epochs):
  model_X.train()
  ypred = model_X(X_train)
  loss = losfn(ypred,Y_train)
  print(f"Loss {loss}")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model_X.eval()
  with inference_mode():
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    testpred = model_X(X_test)
    testloss = losfn(testpred,Y_test)
    test_loss_values.append(testloss)

  if epoch % 20 == 0:
    print(f"epoch {epoch} | Train Loss {loss} | Test Loss {testloss}")
    print(model_X.state_dict())
  with inference_mode():
    mod_cv = model_X(X_test)
plot_predictions(predictions=mod_cv)