import matplotlib.pyplot as plt
import csv

# Initialize lists to store epoch and accuracy values
epochs = []
train_acc = []
train_loss = []
train_psnr = []

# Path to your CSV file
cifar_csv = ['results/Acc_DenseNet121_Cifar10.csv', 
           'results/acc_semantic_combining_0.30_lambda_0.70.csv']

plt.figure(figsize=(10, 6))
color_cnt = 0
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#FF5733', '#33FF57', '#5733FF', '#33FFC3' ]

#cifar
# for snr_ in range (3, 11):
# for snr_ in range (-5, 21, 5):
# for cr in range (1, 10):
epochs = []
train_acc_densenet = []
train_acc_googlenet = []
train_loss = []
train_psnr = []
for file_name in cifar_csv:
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        cnt = 0
        for row in csv_reader:
            if row and ("DenseNet121_Cifar10" in file_name):
                epoch_num = cnt
                cnt += 1
                epochs.append(epoch_num)

                epoch_info = row[0]
                train_accuracy__ = float(epoch_info)/100.0
                train_acc_densenet.append(train_accuracy__)

            if row and ("semantic_combining" in file_name):
                epoch_info = row[0]
                train_accuracy__ = float(epoch_info)
                train_acc_googlenet.append(train_accuracy__)

            # if row and ("psnr" in file_name):
            #     epoch_info = row[0]
            #     train_psnr__ = float(epoch_info)
            #     train_acc.append(train_psnr__)
    
plt.plot(epochs, train_acc_densenet, marker='o')
plt.plot(epochs, train_acc_googlenet, marker = 'o')
# plt.plot(epochs, train_loss, label='cifar - Loss, SNR ' + str(snr_), marker='o')
# plt.plot(epochs, train_psnr, label='cifar - Train PSNR', marker='o')


# Add labels and a legend
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
