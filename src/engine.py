from tqdm.auto import tqdm
import torch


def train(model, data_loader, optimizer, criterion, device, metric, max_norm):
    model.train()
    
    epoch_loss = 0
    epoch_metric = 0
    
    pr_bar = tqdm(data_loader, total=len(data_loader))

    for (x, y) in pr_bar:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        epoch_loss += loss.item()
        _metric = metric(pred, y)
        epoch_metric += _metric

        pr_bar.set_postfix(loss=loss.cpu().data.numpy(), metric = _metric.cpu().data.numpy())

    return epoch_loss / len(data_loader), epoch_metric / len(data_loader)


def evaluate(model, data_loader, criterion, device, metric):
    model.eval()

    epoch_loss = 0
    epoch_metric = 0

    with torch.no_grad():
    
        pr_bar = tqdm(data_loader, total=len(data_loader))
        for (x, y) in pr_bar:  
            x = x.to(device)
            y = y.to(device)

            pred = model(x)           
            loss = criterion(pred, y)

            epoch_loss += loss.item()
            _metric = metric(pred, y)
            epoch_metric += _metric

            pr_bar.set_postfix(loss=loss.cpu().data.numpy(), metric = _metric.cpu().data.numpy())

    return epoch_loss / len(data_loader), epoch_metric / len(data_loader)

def predict(model, data, device):
    model.eval()

    data = data.to(device)

    with torch.no_grad():
        pred = model(data)
    return pred.cpu().data.numpy()