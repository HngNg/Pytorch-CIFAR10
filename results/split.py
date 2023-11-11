import csv
import re

def extract_accuracy(line):
    # Use regular expression to extract the accuracy value
    match = re.search(r'Acc: ([\d.]+)%', line)
    if match:
        return float(match.group(1))
    else:
        return None

def process_csv(input_file, output_file):
    accuracies = []

    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        for row in csv_reader:
            for item in row:
                accuracy = extract_accuracy(item)
                if accuracy is not None:
                    accuracies.append(accuracy)

    # Write accuracies to a new CSV file
    with open(output_file, 'w', newline='') as output_csv:
        csv_writer = csv.writer(output_csv)
        csv_writer.writerow(['Accuracy'])

        for accuracy in accuracies:
            csv_writer.writerow([accuracy])

    print(f'Accuracies written to {output_file}')

# Example usage
input_file_path = 'results/Output_DenseNet121_Cifar10.csv'
output_file_path = 'Acc_DenseNet121_Cifar10.csv.csv'
process_csv(input_file_path, output_file_path)
