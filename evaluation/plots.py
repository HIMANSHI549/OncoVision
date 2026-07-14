import matplotlib.pyplot as plt


class TrainingPlots:

    @staticmethod
    def plot(history, history_fine):

        acc = history.history["accuracy"]

        acc += history_fine.history["accuracy"]

        val_acc = history.history["val_accuracy"]

        val_acc += history_fine.history["val_accuracy"]

        loss = history.history["loss"]

        loss += history_fine.history["loss"]

        val_loss = history.history["val_loss"]

        val_loss += history_fine.history["val_loss"]

        epochs = range(1, len(acc)+1)

        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)

        plt.plot(epochs, acc)

        plt.plot(epochs, val_acc)

        plt.title("Accuracy")

        plt.legend(["Train","Validation"])

        plt.subplot(1,2,2)

        plt.plot(epochs, loss)

        plt.plot(epochs, val_loss)

        plt.title("Loss")

        plt.legend(["Train","Validation"])

        plt.tight_layout()

        plt.show()