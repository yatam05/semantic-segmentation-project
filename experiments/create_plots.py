import matplotlib.pyplot as plt
import numpy as np

i = 1
epochs = []
for i in range(200):
    epochs.append(i)

train = np.loadtxt("checkpoints/train_accuracy.txt")
val = np.loadtxt("checkpoints/validation_accuracy.txt")

plt.title('Training and Validation mIoU')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.plot(epochs, train, label='Training mIoU')
plt.plot(epochs, val, label='Validation mIoU')
plt.legend()
plt.show()