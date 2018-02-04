import re
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Add as many as filenames possible to logfile_paths. Each graph will plot these file curves
file1 = "../logs/2018-01-21_22-58-39-resnet18-train-0.22387.log"
file2 = "../logs/2018-01-22_10-20-27-resnet18-train-0.53539.log"
logfile_paths = [file1, file2]
logfile_graph_labels = ["A", "B"]

# Dictionary to store all log messages for each file
logs = {}

# Parse each logfile and read list of messages into dictionary logs
for i, filename in enumerate(logfile_paths):
    f = open(filename, "r")
    lines = f.read().split("\n")
    messages = []
    for line in lines:
        if re.match("\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ", line):
            messages.append(line)
        else:
            messages[-1] += line
    # Map messages to file basename only from the whole path.
    logs[logfile_graph_labels[i]] = messages

# Extract training loss for each log file
ALL_TRAIN_QWK={}
ALL_VAL_QWK={}
ALL_TRAIN_LOSS={}
ALL_VAL_LOSS={}
ALL_CUMULATIVE_TRAINING_TIME = {}
ALL_TRAINING_TIME_PER_EPOCH = {}

numberExtractPattern = re.compile(r'[+-]?\d+(?:\.\d+)?')

def getTrainingLossForEachEpoch(messagesList):
    TRAIN_QWK = []
    TRAIN_LOSS = []
    for message in messagesList:
        if "TRAIN   QWK" in message:
            TRAIN_QWK.append(float(numberExtractPattern.findall(message)[9]))
            TRAIN_LOSS.append(float(numberExtractPattern.findall(message)[10]))
    return  TRAIN_QWK, TRAIN_LOSS

def getValidationLossForEachEpoch(messagesList):
    VALIDATION_QWK = []
    VALIDATION_LOSS = []
    for message in messagesList:
        if "VAL     QWK" in message:
            VALIDATION_QWK.append(float(numberExtractPattern.findall(message)[9]))
            VALIDATION_LOSS.append(float(numberExtractPattern.findall(message)[10]))
    return VALIDATION_QWK, VALIDATION_LOSS

def getTrainingTimeForEachEpoch(messagesList):
    CUMULATIVE_TRAINING_TIME = []
    TRAINING_TIME_PER_EPOCH = []
    prevTrainingTime = 0
    for message in messagesList:
        if " - Training Time - " in message:
            CUMULATIVE_TRAINING_TIME.append(float(numberExtractPattern.findall(message)[8]))
            TRAINING_TIME_PER_EPOCH.append(float(numberExtractPattern.findall(message)[8]) - prevTrainingTime)
            prevTrainingTime = CUMULATIVE_TRAINING_TIME[-1]
    return CUMULATIVE_TRAINING_TIME, TRAINING_TIME_PER_EPOCH

for label in logfile_graph_labels:
    ALL_TRAIN_QWK[label], ALL_TRAIN_LOSS[label] = getTrainingLossForEachEpoch(logs[label])
    ALL_VAL_QWK[label], ALL_VAL_LOSS[label] = getValidationLossForEachEpoch(logs[label])
    ALL_CUMULATIVE_TRAINING_TIME[label], ALL_TRAINING_TIME_PER_EPOCH[label] = getTrainingTimeForEachEpoch(logs[label])

figureCount=1

def plotGraphOfMultipleLogFiles(logFileLabels, yValues, xLabel="Epoch", yLabel="Training Loss"):
    global figureCount
    plt.figure(figureCount)
    figureCount += 1
    for label in logfile_graph_labels:
        numEpochs = len(yValues[label])
        epochList = list(range(1,numEpochs+1))
        plt.plot(epochList, yValues[label], label=label)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()

plotGraphOfMultipleLogFiles(logfile_graph_labels, ALL_TRAIN_LOSS, xLabel="Epoch", yLabel="Training Loss")
plotGraphOfMultipleLogFiles(logfile_graph_labels, ALL_VAL_LOSS, xLabel="Epoch", yLabel="Validation Loss")
plotGraphOfMultipleLogFiles(logfile_graph_labels, ALL_TRAIN_QWK, xLabel="Epoch", yLabel="Training QWK")
plotGraphOfMultipleLogFiles(logfile_graph_labels, ALL_VAL_QWK, xLabel="Epoch", yLabel="Validation QWK")
plotGraphOfMultipleLogFiles(logfile_graph_labels, ALL_CUMULATIVE_TRAINING_TIME, xLabel="Epoch", yLabel="Total Time")
plotGraphOfMultipleLogFiles(logfile_graph_labels, ALL_TRAINING_TIME_PER_EPOCH, xLabel="Epoch", yLabel="Time per Epoch")
plt.show()
