from torchvision import transforms
import torch
import numpy as np

# Helper functions

def class_weights(labels, device, beta=0.99):
    """
    Calculates weights for cross-entropy loss, based on effective number of samples in training data.
    Addresses problems of class imbalance in training.
    Args:
        labels: list of labels from training data
    """
    # Find all unique labels
    classes = np.unique(labels)
    
    # Find number of samples per class
    samples_per_class = []
    for cls in classes:
        samples_per_class.append(sum(labels==cls))
        
    # Convert number of samples into weights
    effective_num = (1-0 - beta) / (1.0 - np.power(beta, samples_per_class))
    weights = effective_num / np.sum(effective_num)
    
    # Convert to tensor
    weights = torch.tensor(weights).float().to(device)
    
    return weights

def augment_batch(batch):
    """
    Applies random transformations to images in the training batch and adds them to the batch.
    Prevents over-fitting.
    Args:
        batch: array of training data
    """
        
    # Convert batch to PyTorch tensors
    batch = torch.Tensor(np.transpose(batch, (0, 3, 1, 2)))
    
    # Apply transforms to each image in batch
    augmented_data = torch.empty_like(batch)
    for i in range(batch.shape[0]):
        
        # Define transforms (randomised for each iteration)
        tsfm = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.RandomPerspective(0.05),
            transforms.RandomErasing(0.25, (0.04, 0.04))
        )

        # Transform image
        img = tsfm(batch[i])
        augmented_data[i] = img
    
    # Combine original images and augmented images
    batch = torch.cat((batch, augmented_data))

    return batch

def get_train_labels(batch_labels):
    """
    Doubles the length of batch_labels, to match the augmented training data, and converts to a tensor.
    """
    
    labels = []
    [labels.append(x) for x in batch_labels]
    [labels.append(x) for x in batch_labels]
    labels = [0 if x==-1 else 1 for x in labels]
    labels = torch.ravel(torch.tensor(labels))
    return labels

def get_val_labels(batch_labels):
    """
    Converts -1 values to 0, and list to tensor.
    """
    
    labels = [0 if x==-1 else 1 for x in batch_labels]
    labels = torch.ravel(torch.tensor(labels))
    return labels

def get_validation_loss(dataset, batch_size, device):
    """
    Gets batches from the validation set (easier to fit in memory) and calculates prediction loss
    """
    X_val_batches = dataset.get_batches(batch_size//2, validate=True)
    Y_val_batches = dataset.get_batch_labels(batch_size//2, validate=True)
    val_batch_loss = []
    for val_batch, val_labels in zip(X_val_batches, Y_val_batches):
        current_batch_loss, _ = model.validation_step(torch.Tensor(np.transpose(val_batch, (0, 3, 1, 2))).to(device),
                        get_val_labels(val_labels).to(device))
        val_batch_loss.append(current_batch_loss.ravel().item())
    val_batch_loss = np.mean(val_batch_loss)

    return val_batch_loss

def plot_training_validation_loss(training_loss, val_loss, filepath):
    """
    Plots training and validation loss and saves png to file
    """
    # Plot training loss
    plt.plot([i for i in range(len(training_loss))], np.squeeze(training_loss))
    
    # Overlay validation loss at intervals of 10 epochs
    try:
        plt.plot([i for i in range(0, len(training_loss), 10)], np.squeeze(val_loss))
    except:
        plt.plot([i for i in range(0, len(training_loss)-10, 10)], np.squeeze(val_loss))
        
    # Add labels
    plt.legend(['Training loss', 'Validation loss'], frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    ax = plt.gca()
    ax.set_ylim(bottom=0)
    
    # Save
    plt.savefig(filepath)
    
def final_validation(validation_dataset, batch_size, device):
    """
    Runs model on whole validation set (in batches to fit in memory) and returns lists of ground-truth labels and predicted labels
    """
    # Break down validation set into batches
    X_val_batches = validation_dataset.get_batches(BATCH_SIZE//2, validate=True)
    Y_val_batches = validation_dataset.get_batch_labels(BATCH_SIZE//2, validate=True)
    
    # Get labels
    predicted_labels = []
    true_labels = []
    for val_batch, val_labels in zip(X_val_batches, Y_val_batches):
            _, pred = model.validation_step(torch.Tensor(np.transpose(val_batch, (0, 3, 1, 2))).to(DEVICE),
                                                get_val_labels(val_labels).to(DEVICE))
            [true_labels.append(0) if x==-1 else true_labels.append(1) for x in val_labels]
            [predicted_labels.append(p) for p in pred]
    
    return true_labels, predicted_labels