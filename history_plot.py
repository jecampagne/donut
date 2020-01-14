import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='loss-plot')
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--log', default=False,action='store_true')

args = parser.parse_args()
data = np.load(args.file)

train_loss = data[0]
test_loss = data[1]
test_psnr = data[2]

fig, ax1 = plt.subplots()
ax1.plot(train_loss, label="train", color='tab:blue')
ax1.plot(test_loss, label="test", color='tab:orange')
if args.log:
    ax1.set_yscale("log")
ax1.set_xlabel('epoch')
ax1.set_ylabel('Cross Entropy loss')
ax1.tick_params(axis='y')
ax1.grid()
ax1.legend()


ax2 = ax1.twinx()
ax2.plot(test_psnr, color='tab:green')
ax2.set_ylabel('PSNR')
ax2.tick_params(axis='y', labelcolor='tab:green')

plt.title(args.tag,fontsize=16)
fig.tight_layout()

plt.savefig('history_'+args.tag+'.png')
plt.show()
