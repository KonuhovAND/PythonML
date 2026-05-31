import matplotlib.pyplot as plt
import random as rd

x = [i for i in range(100)]
y1 = [rd.randint(0,100) for _ in range(100)]
plt.style.use("ggplot")
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x,y1,linestyle=":",marker="*",color="red")
plt.show()