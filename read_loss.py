import re
import numpy as np

loss=[]
loss1=[]
loss2_3=[]
i=0
with open("modeltestLog.log", "r") as f:
    for line in f.readlines():
        i=i+1
        if i%2==0:
            continue
        line = line.strip('\n')
        print(line)
        e,l,ll,l1,l2,l3,ls=re.findall(r"[-+]?\d*\.\d+|\d+", line)
        print(l,l1,ls)
        loss.append(float(l))
        loss1.append(float(l1))
        loss2_3.append(float(ls))
loss=loss[:600]
loss1=loss1[:600]
loss2_3=loss2_3[:600]
loss=np.array(loss)
loss1=np.array(loss1)
loss2_3=np.array(loss2_3)
print(loss)
print(loss1)
print(loss2_3)

print('loss',len(loss))
import matplotlib.pyplot as plt
import torch
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

x=np.arange(1,601,1)

fig = plt.figure(figsize = (7,5))
ax1 = fig.add_subplot(1, 1, 1)

pl.plot(x,loss,'r-',label=u'loss')
p2 = pl.plot(x, loss1,'g-', label = u'loss1')
pl.legend()
p3 = pl.plot(x,loss2_3, 'b-', label = u'loss2_3')
pl.legend()
pl.xlabel(u'epoch')
pl.ylabel(u'loss')
plt.title('loss for PPIformer in training')
plt.show()