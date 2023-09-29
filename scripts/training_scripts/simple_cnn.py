import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# Define model
class simple_cnn(nn.Module):
    
    def __init__(self, batch_size=512, layer1_size=32, layer2_size=64, kernel_size=[3, 3]):
        
        super(simple_cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=layer1_size, kernel_size=kernel_size[0])
        self.conv2 = nn.Conv2d(in_channels=layer1_size, out_channels=layer2_size, kernel_size=kernel_size[1]) #ER WHAT IS THE CORRECT SIZE?
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self.loss_function = nn.CrossEntropyLoss(weight=torch.unsqueeze(torch.tensor([0.5,0.5]), 1))
        self.optimizer = ()

    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

    def training_step(self, train_batch, train_labels):
            
        # Initialise list to store losses
        batch_loss = []
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        y = self.forward(train_batch)
        
        # Calculate loss
        loss = self.loss_function(y, train_labels)
        
        # Backward pass
        loss.backward()
        
        # Update gradient
        self.optimizer.step()
        
        # Store losses
        batch_loss.append(np.array(loss.item()))
        
        return batch_loss
    
    def validation_step(self, val_batch, val_labels):
        
        self.eval()
        
        with torch.no_grad():
            output = model(val_batch)
            loss = self.loss_function(output, val_labels)
            _, pred = torch.max(output, 1)
            pred = pred.cpu().detach().numpy()
            
        return loss, pred