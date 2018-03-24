import matplotlib.pyplot as plt


plt.figure(num = "graph")
plt.ion()
axes = plt.gca()
axes.set_xlim(0,5)
axes.set_ylim(0,10)
while True:
    for i in range(10):
        axes = plt.gca()
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 10)
        plt.bar(0, i, color = "0.5")
        plt.pause(0.05)
        plt.cla()

    for i in range(10):
        axes = plt.gca()
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 10)
        plt.bar(0, 10-i, color = "0.5")
        plt.pause(0.05)
        plt.cla()

plt.show()
