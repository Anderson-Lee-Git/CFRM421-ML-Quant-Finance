import tensorflow as tf
from autoencoder import AutoEncoder

def train_step(x, model, optimizer, loss_fn):
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        pred = model(x)
        # Loss value for this batch.
        loss_value = loss_fn(x, pred)
    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss_value

def validate_step(x, model, loss_fn):
    val_pred = model(x)
    loss_value = loss_fn(x, val_pred)
    return loss_value

def train_autoencoder(train_dataloader, val_dataloader, epochs, optimizer=None, lr_schedule=None, pretrain_path=None, loss_fn=tf.keras.losses.MeanSquaredError(), args: dict = None):
    if lr_schedule is None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args['learning_rate'],
            decay_steps=10000,
            decay_rate=0.9)
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    best_val_loss = None
    lr = args['learning_rate']
    alpha = args['alpha']
    best_path = f'model_best_lr_{lr}_alpha_{alpha}'
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        print(f'Start of {epoch + 1} epoch')
        train_loss = 0
        for x in train_dataloader:
            loss_value = train_step(x, model, optimizer, loss_fn)
            train_loss += loss_value

        print(f'Train MSE = {train_loss}')

        val_loss = 0
        for val_x in val_dataloader:
            loss_value = validate_step(val_x, model, loss_fn)
            val_loss += loss_value

        print(f'Validation MSE = {val_loss}')
        print()

        val_loss = val_loss / len(val_dataloader)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(best_path)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    model.save('model_latest')
    return train_losses, val_losses

if __name__ == '__main__':
    n = 20
    d = full_data.shape[1] - 1
    epochs = 60

    X_train = fold_data[0]['X_train']
    X_val = fold_data[0]['X_val']
    # Initialize the dataset
    batch_size = 2048
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train))
    train_dataloader = train_dataset.batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val))
    val_dataloader = val_dataset.batch(batch_size)
    args = {
        'learning_rate': 1e-3,
        'alpha': 0.01
    }
    lr = args['learning_rate']
    alpha = args['alpha']
    # model = keras.models.load_model("after_300_epochs")
    model = AutoEncoder(64, 20, n, d, l2_alpha=args['alpha'])
    # Learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args['learning_rate'],
        decay_steps=10000,
        decay_rate=0.9)
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # Instantiate a loss function
    loss_fn = tf.keras.losses.MeanSquaredError()
    train_history, val_history = train_autoencoder(train_dataloader, val_dataloader, epochs, args=args)