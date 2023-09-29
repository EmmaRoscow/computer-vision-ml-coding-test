import torch.nn as nn
import torch
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score
from simple_cnn import simple_cnn
from helpers import class_weights, augment_batch, get_train_labels, get_val_labels, get_validation_loss, plot_training_validation_loss, final_validation
import read_data as rd


def train_model_main():
    
    # Set up
    SEED = 2911
    torch.manual_seed(SEED)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = torch.device(DEVICE)
    BATCH_SIZE = 512
    
    # Initialise model
    model = simple_cnn(batch_size=BATCH_SIZE).to(DEVICE)
    
    # Load data
    dataset_path='/data/scaled/tensorflow_datasets/'
    train_dataset = rd.Dataset(dataset_path=dataset_path)
    validation_dataset = rd.Dataset(dataset_path=dataset_path)
    
    # Specify loss function
    Y_train = train_dataset.Y_train
    w = class_weights(Y_train)
    model.loss_function = nn.CrossEntropyLoss(weight=w)
    
    # Specify optimizer
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Load batches for training
    X_batches = train_dataset.get_batches(BATCH_SIZE//2, validate=False)
    Y_batches = train_dataset.get_batch_labels(BATCH_SIZE//2, validate=False)
    
    # Train the model
    training_loss = []
    val_loss = []
    batch_count = 0
    for batch, labels in tqdm(zip(X_batches, Y_batches), desc='Training model...'):
        
        # Augment data
        train_batch = augment_batch(batch)
        
        # Get labels
        train_labels = get_train_labels(labels)
            
        # Train
        batch_loss = model.training_step(train_batch.to(DEVICE), train_labels.to(DEVICE))
        training_loss.append(batch_loss[0])
        
        # Validate every 10 epochs
        if batch_count % 10 == 0:
            val_loss.append(get_validation_loss(validation_dataset, BATCH_SIZE))
        
        batch_count += 1
        
    # Save model
    print("Saving model...")
    model_path = '/models/simple_cnn'
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), model_path+'/model_'+now)
    
    # Plot training and validation loss
    filepath = model_path+'/model_'+now+'training_val_loss.png'
    plot_training_validation_loss(training_loss, val_loss, filepath)
    
    # Get final validation score
    true_labels, predicted_labels = final_validation(validation_dataset, BATCH_SIZE, DEVICE)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print("Weighted F1 score: {f1:.4f}".format(f1=f1))
