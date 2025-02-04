import torch
from pathlib import Path
from tqdm import tqdm

def train(epoch, train_loader, val_loader, optimizer, model, device, save_dir, writer=None):
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}, Loss : ')
    loss_list = []
    model.train()
    num_iter = epoch*len(train_loader)
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        loss_list.append(loss)
        optimizer.step()
        pbar.set_description(f'Epoch {epoch}, Loss : {loss:.5f}')
        if writer is not None:
            writer.add_scalar('training loss per batch', loss, num_iter)
        if (val_loader is not None) and (num_iter % 1000 == 0):
            val_acc = val(val_loader, model, device)
            save(model, num_iter, 'val_iter', save_dir)
            if writer is not None:
                writer.add_scalar('validation acc per 1000 iter', val_acc, num_iter)
        num_iter += 1
        
    if val_loader is None:
        save(model, epoch, 'train', save_dir)
    else:
        save(model, epoch, 'val', save_dir)

    if writer is not None:
        epoch_loss = sum(loss_list)/len(train_loader)
        writer.add_scalar('training avg loss per epoch', epoch_loss, epoch)


def val(val_loader, model, device):
    pbar = tqdm(val_loader, desc='Evaluation Acc')
    acc_list = []
    model.eval()
    num_iter = 0
    for batch in pbar:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) 
            outputs = model(input_ids, attention_mask=attention_mask)
            num_iter += 1
            acc_list.append(torch.sum(torch.max(outputs[0], dim=1)[1].cpu() == batch['labels'])/len(batch['labels']))
            pbar.set_description(f'Evaluation Acc : {sum(acc_list)/num_iter:.5f}')
    model.train()
    return sum(acc_list)/num_iter #total acc

def test(test_loader, model, device):
    info_list = []
    model.eval()
    for batch in tqdm(test_loader, desc='Make Submission'):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            info_list.append(torch.max(outputs[0], dim=1)[1])
    info_list = torch.cat(info_list)
    return info_list

def save(model, epoch, mode, save_dir):
    save_dir = Path(save_dir)
    try:
        if not save_dir.exists():
            save_dir.mkdir()
    except:
        print(f'Error: Creating directory. {save_dir}')

    file_name = f'{mode}_{epoch}.pt'
    new_dir = save_dir / file_name
    torch.save(model.state_dict(), new_dir)
    print(f'Save your model as {new_dir}')