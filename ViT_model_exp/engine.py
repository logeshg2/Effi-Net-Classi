'''
    Modular code -> train function
'''

import torch
from torch import nn
from tqdm.auto import tqdm

# train_step
def train_step(model, loss_fn, optimizer, train_dataloader, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch, (X, y) in tqdm(enumerate(train_dataloader)):
        # y_pred = torch.softmax(pred_logits,dim=1).argmax(dim=1)
        X, y = X.to(device), y.to(device)
        # forward pass
        y_logits = model(X)
        y_pred_class = torch.softmax(y_logits, dim=1).argmax(dim=1)
        # compute loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        # compute accuacy
        train_acc += (y_pred_class == y).sum().item() / len(y_logits)
        # optimizer
        optimizer.zero_grad()   
        # backprop
        loss.backward()
        # update parameters
        optimizer.step()
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)
    return train_loss, train_acc

# test step
def test_step(model, loss_fn, test_dataloader, device):
    model.eval()
    test_loss = 0; test_acc = 0
    if torch.inference_mode():                       
        for batch, (X, y) in enumerate(test_dataloader):
            # put data into the device
            X, y = X.to(device), y.to(device)
            # forward
            logits_test = model(X)
            y_pred_class = torch.softmax(logits_test, dim=1).argmax(dim=1)
            # compute loss and accuracy
            loss = loss_fn(logits_test, y)
            test_loss += loss.item()
            test_acc += (y_pred_class == y).sum().item() / len(logits_test)
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)
    return test_loss, test_acc


# main train function
def train_model(model,
                loss_fn,
                optimizer,
                train_dataloader,
                test_dataloader,
                epochs,
                device
                ):
    
    for epoch in tqdm(range(epochs)):
        print(f"------ Epoch {epoch} ------")
        
        # train step for each epoch
        train_loss, train_acc = train_step(model, loss_fn, optimizer, train_dataloader, device)
        # test step for each epoch
        test_loss, test_acc = test_step(model, loss_fn, test_dataloader, device)

        print(f"Train loss: {train_loss:.2} | Train Accuracy: {(round(train_acc, 4) * 100)} %")
        print(f"Test loss: {test_loss:.2} | Test Accuracy: {(round(test_acc, 4) * 100)} %")
    

# main eval function
def eval_model(model, loss_fn, test_dataloader, device):
    # evaluation of model script
    # same a test_step()
    eval_loss, eval_acc = test_step(
        model=model,
        loss_fn=loss_fn,
        test_dataloader=test_dataloader,
        device=device
    )
    print(f"Evaluation:\nEval loss: {eval_loss:.2} | Eval Accuracy: {(round(eval_acc, 4) * 100)} %")