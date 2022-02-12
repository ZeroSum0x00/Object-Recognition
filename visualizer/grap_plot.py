import matplotlib.pyplot as plt


def plot_training(history, path_acc, path_loss):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'b', label="Training set accuracy")
    plt.plot(epochs, val_acc, 'r', label="Test set accuracy")
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(path_acc)

    plt.figure()
    plt.plot(epochs, loss, 'b', label="Training set loss")
    plt.plot(epochs, val_loss, 'r', label="Test set loss")
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path_loss)


def plot_training_history(history, path):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='--', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')
    
    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Save the image
    plt.savefig(path)

    # Ensure the plot shows correctly.
    plt.show()