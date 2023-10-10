import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable

hp = {
    'epochs': 10,
    'LR': 0.01,
    'batch_size': 128,
    'validation_steps': 100,
}

def train(num_epochs, cnn, loaders, validation_steps):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % hp.validation_steps == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass
        pass
    pass

class CNN(nn.Module):
    def __init__(self, hp):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization


if __name__ == "__main__":

    # check GPU usage.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device being used: {device}\n')

    # load datasets
    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = ToTensor(),
        download = True,
    )
    test_data = datasets.MNIST(
        root = 'data',
        train = False,
        transform = ToTensor()
    )
    print(f'Training data: {train_data}\n')
    print(f'Test data: {test_data}\n')
    
    # data loaders
    loaders = {
        'train' : DataLoader(train_data,
                        batch_size=hp['batch_size'],
                        shuffle=True,
                        num_workers=1),
        'test'  : DataLoader(test_data,
                        batch_size=hp['batch_size'],
                        shuffle=True,
                        num_workers=1),
    }
    print(f'Data loaders: {loaders}\n')

    # build model
    cnn = CNN(hp)
    print(f'Model:\n {cnn}\n')

    # loss
    loss_func = nn.CrossEntropyLoss()
    print(f'Loss function: {loss_func}\n')

    # optimizer
    optimizer = optim.Adam(cnn.parameters(), lr = hp['LR'])
    print(f'Optimizer: {optimizer}\n')

    num_epochs = hp['epochs']

    # train model
    train(num_epochs, cnn, loaders, hp['validation_steps'])

