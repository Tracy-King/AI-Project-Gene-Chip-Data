import numpy as np

trX = []
trY = []

row = 22283
col = 5895
num = 0
percent = 0

fx = open('../microarray.original.txt','r')
fy = open('../microarray.transpose1.txt','w')
fx.readline()

for i in range(0,22283):
    line_x = fx.readline().split('\t')[:-1]
    #if float(line_x[1])<10:
     #       continue
    num+=1
    line_x = line_x[1:]
    line_x = list(map(float,line_x))
    trX.append(line_x)
    if(i % 222 == 0):
        percent = percent + 1
        print("Input %:",percent)

print(num)
print(len(trX))
print("Transposing...")
data = np.array(trX).T
print("Transposing Finished")
print(len(data))

percent = 0

for i in range(0,len(data)):
    for j in range(0,len(data[i])):
        fy.write(str(data[i][j])+'\t')
    fy.write('\n')
    if(i % 58 == 0):
        percent = percent + 1
        print("Writing %:",percent)
    
fx.close()
fy.close()
