import numpy as np
import torch

def fit(net, config, train_loader, val_loader, device, logger):

  from tqdm import tqdm
  
  optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
  criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(0, config['epochs']):
    n_train_acc, n_val_acc = 0, 0
    train_loss, val_loss = 0, 0
    n_train, n_val = 0, 0

    # 学習
    net.train()

    for inputs, labels in tqdm(train_loader):
      
      train_batch_size = len(labels)
      n_train += train_batch_size

      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      outputs = net(inputs)

      loss = criterion(outputs, labels)

      loss.backward()

      optimizer.step()

      predicted = torch.max(outputs, 1)[1]

      train_loss += loss.item() * train_batch_size
      n_train_acc += (predicted == labels).sum().item()

    # 検証
    net.eval()

    for inputs_val, labels_val in val_loader:
      val_batch_size = len(labels_val)
      n_val += val_batch_size

      inputs_val = inputs_val.to(device)
      labels_val = labels_val.to(device)

      outputs_val = net(inputs_val)

      loss_val = criterion(outputs_val, labels_val)

      predicted_val = torch.max(outputs_val, 1)[1]

      val_loss += loss_val.item() * val_batch_size
      n_val_acc += (predicted_val == labels_val).sum().item()

  

    train_acc = n_train_acc / n_train
    val_acc = n_val_acc / n_val

    avg_train_loss = train_loss / n_train
    avg_val_loss = val_loss / n_val

    logger.log({"epoch": epoch+1, "train_loss": avg_train_loss, "train_acc": train_acc, "val_loss": avg_val_loss, "val_acc": val_acc})


  
    