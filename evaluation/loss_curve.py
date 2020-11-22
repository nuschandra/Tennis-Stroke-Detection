import numpy as np
import matplotlib.pyplot as plt

loss_list = [0.3427, 1.0110, 1.0798, 0.0956, 0.1151, 1.2670, 0.1056, 0.1360, 0.2326, 1.3486, 0.0056, 0.0472, 0.0210, 0.0946, 0.0287]

x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
plt.plot(x_axis, loss_list)
plt.savefig('F:/Projects/YOWO-master/evaluation/loss_curve.png')
plt.xlabel('epoch')
plt.ylabel('loss_value')
plt.title('Loss curve')
# plt.legend(shadow=True)
plt.grid()
plt.show()
