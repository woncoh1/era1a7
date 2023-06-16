
import torch


def train_step(
    device:torch.device,
    loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    criterion:torch.nn.functional,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LRScheduler,
    epoch:int,
    onecyclelr:bool=False,
) -> tuple[float, float]:
    model.train()
    train_loss = 0
    correct = 0
    processed = 0
    epoch += 1
    pbar = tqdm(loader)
    for batch, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if onecyclelr: scheduler.step()
        train_loss += loss.item()
        correct += count_correct_predictions(pred, y)
        processed += len(X)
        pbar.set_description(desc=(
            f"Epoch = {epoch}, "
            f"Batch = {batch}, "
            f"Loss = {loss.item():0.4f}, "
            f"Accuracy = {correct/processed:0.2%}"
        ))
    if not onecyclelr: scheduler.step()
    n = len(loader.dataset)
    train_loss /= n
    train_acc = correct / n
    print(
        f"Train: "
        f"Loss = {train_loss:0.5f}, "
        f"Accuracy = {train_acc:0.2%}, "
        f"Epoch = {epoch}"
    )
    return train_loss, train_acc


def test_step(
    device:torch.device,
    loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    criterion:torch.nn.functional,
) -> tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            test_loss += loss.item()
            correct += count_correct_predictions(pred, y)
    n = len(loader.dataset)
    test_loss /= n
    test_acc = correct / n
    print(
        f"Test : "
        f"Loss = {test_loss:.5f}, "
        f"Accuracy = {test_acc:.2%}\n"
    )
    return test_loss, test_acc


def train(
    device:torch.device,
    train_loader:torch.utils.data.DataLoader,
    test_loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    criterion:torch.nn.functional,
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LRScheduler,
    epochs:int,
) -> dict[str, list[float]]:
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            device,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
        )
        test_loss, test_acc = test_step(
            device,
            test_loader,
            model,
            criterion,
        )
        scheduler.step()
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc*100)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc*100)
    print('Learning rate =', params_optimizer['lr'])
    return results


def count_correct_predictions(
    predictions:torch.Tensor,
    labels:torch.Tensor,
) -> int:
    return predictions.argmax(dim=1).eq(labels).sum().item()


def find_lr(
    device:torch.device,
    train_loader:torch.utils.data.DataLoader,
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    init_value:float=1e-8,
    final_value:float=10.,
    beta: float=0.98,
) -> tuple[list[float], list[float]]:
    """https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html"""
    trn_loader = train_loader
    net = model
    num = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    lrs = []
    for data in trn_loader:
        batch_num += 1
        # Get the loss for this mini-batch of inputs/outputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # Compute the smoothed loss
        avg_loss = beta*avg_loss+(1-beta)*loss.item()
        smoothed_loss = avg_loss/(1-beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4*best_loss:
            return lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        lrs.append(lr)
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return lrs, losses