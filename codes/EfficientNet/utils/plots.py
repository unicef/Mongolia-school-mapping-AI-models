from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_roc_curve(groundtruths, predictions, pos_label):
    
    fpr, tpr, thresholds = roc_curve(groundtruths, 
                                     predictions, 
                                     pos_label=pos_label)
    roc_auc = auc(x=fpr, y=tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, 
             color='darkorange',
             lw=lw, 
             label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_stats(xticks, train_loss, valid_loss, train_acc, valid_acc):
    
    # Set matplotlib default plot size
    plt.rcParams["figure.figsize"] = [7, 5]

    # Loss
    plt.title("Train and Validation Loss")
    plt.plot(xticks, train_loss, label="Train Loss")
    plt.plot(xticks, valid_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    # Accuracy
    plt.title("Validation Accuracy")
    plt.plot(xticks, train_acc)
    plt.plot(xticks, valid_acc)
    plt.show()