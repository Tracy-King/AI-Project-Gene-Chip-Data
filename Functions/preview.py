f1 = open("microarray.regular.dim2-22283.txt", "r")
f2 = open("microarray.preview2.txt","w")

for i in range(10):
    s = f1.readline()
    f2.write(s)
    print(i)

f1.close()
f2.close()
print("Success")
