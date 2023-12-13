label_prefix = 'Eve' if 'Eve' in file_name else 'Bob'
    plt.plot(epochs, train_acc_densenet, label=f'{label_prefix} - Accuracy, SNR {snr}', marker=