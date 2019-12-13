import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='loss-plot')
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--tag', type=str, required=True)

args = parser.parse_args()
data = np.load(args.file)

plt.plot(data, label="train")
plt.xlabel('epoch')
plt.ylabel('Cross Entropy loss')
plt.legend()
plt.grid()

plt.savefig('history_'+args.tag+'.png')
plt.show()
