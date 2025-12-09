import torch.nn.functional as F
import torch

def training_loop_signature(model, train_loader, val_loader, loss_fcn, optimizer, scheduler, num_epochs, threshold, device):
    best_val_acc = 0.0
    best_state_dict = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            anchor, positive, negative = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            z_a, z_p, z_n = model(anchor, positive, negative, triplet_bool=True)
            loss = loss_fcn(z_a, z_p, z_n)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * anchor.size(0)
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                anchor, positive, negative = batch
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                z_a, z_p, z_n = model(anchor, positive, negative, triplet_bool=True)
                loss = loss_fcn(z_a, z_p, z_n)
                val_loss += loss.item() * anchor.size(0)
                
                d_ap = F.pairwise_distance(z_a, z_p)
                d_an = F.pairwise_distance(z_a, z_n)
                
                genuine_correct = (d_ap < threshold)
                fake_correct = (d_an >= threshold)
                batch_correct = (genuine_correct & fake_correct).sum().item()
                
                correct += batch_correct
                total += anchor.size(0)
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.4f}")
        
        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model