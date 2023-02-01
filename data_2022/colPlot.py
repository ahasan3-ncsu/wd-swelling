import matplotlib.pyplot as plt

with open('eureka.out', 'r') as f:
    jar = f.readlines()

gr = []
qoi = [[] for i in range(6)]
for line in jar:
    tmp = line.split()
    gr.append(float(tmp[0]))
    for i in range(6):
        qoi[i].append(float(tmp[i+1]))

for i in range(6):
    plt.plot(gr, qoi[i], label=f'{i}')

plt.legend()
plt.show()
#plt.savefig('eureka.pdf')
