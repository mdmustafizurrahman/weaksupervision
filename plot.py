f = open("result_CNN.txt")
print f
tmplist = []
for lines in f:
    values = lines.split(",")


y_val = []
x_val = []
inc = 1000
for val in values:
    print val
    if val == "":
        continue
    y_val.append(float(val))
    x_val.append(inc)
    inc = inc + 1000
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

plt.ylabel('accuracy')
#plt.xticks(x_val)
plt.plot(x_val, y_val)
plt.xlabel("Training Size")
plt.show()
