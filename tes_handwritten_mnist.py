import tensorflow as tf
from matplotlib import pyplot as plt


def handwritten():
    mnist = tf.keras.datasets.mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    training_images = x_train / 255.0
    test_images = x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    #callbacks = myCallback()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    process = model.fit(training_images, y_train, epochs=20, validation_data = (test_images, y_test))

    acc = process.history['accuracy']
    val_acc = process.history['val_accuracy']
    loss = process.history['loss']
    val_loss = process.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    if __name__ == '__main__':
        model = handwritten()
        model.save("model-handwritten.h5")