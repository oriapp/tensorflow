import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Numbers
X = tf.range(-100, 100, 4)
y = X + 10
X_train = X[:40]
y_train = y[:40]


X_test = X[40:]
y_test = y[40:]

tf.random.set_seed(42)

# Modeling
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1, name="output_layer")
], name="model_1")

# Compiling
model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["mae"]
)

# Fitting
model.fit(X_train, y_train, epochs=100, verbose=0)

y_pred = model.predict(y_test)

def predicts(train_data = X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_pred):
    """
    Plots trainign data, test data, and compares predictions to ground truth labels
    """
    plt.figure(figsize=(10, 7))
    # Plot trainign data in blue
    plt.scatter(train_data, train_labels, c="blue", label="Training data")
    # Plot testing data in green
    plt.scatter(test_data, test_labels, c="green", label="Testing data")
    # Plot models predictions in red
    plt.scatter(test_data, predictions, c="red", label="Predictions")
    # Show the legend
    plt.legend()
    return plt.show()

predicts()
