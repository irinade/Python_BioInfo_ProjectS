import matplotlib.pyplot as plt
import random
import math
import numpy as np

file = open('data1.txt','r')
A = file.readline()
A = A.rstrip("\n\r")


x = [i for i in range(200)] 
y = [random.uniform(4,12) for i in range(200)]

#for i in range(200):
#	y.append(random.uniform(4,12))

plt.plot(x,y)
plt.xlabel('x : list of 200 integer values regularly spaced from 0 to 199')
plt.ylabel('y : list of 200 real numbers randomly \n generated in the interval [4, 12]')
plt.title('Fonction : y = alea(x)')
plt.show()


val_base = {"A":1,"T":2,"C":3,"G":4}
x = []
y = []

for i in range(len(A)):
	x.append(i)
	for key,val in val_base.items():
		if A[i] == key:
			y.append(val)


plt.plot(x,y)
plt.xlabel('x : Position in the DNA sequence')
plt.ylabel('y : Values assigned to each of the baise pair \n "A":1 ; "T":2 ; "C":3 ; "G":4')
plt.title('Fonction : y = f(x) : Position of the base pair in the DNA sequence, \n Evolution of y values as a function of the position x in the sequence ')
plt.show()

x = np.arange(-2*math.pi,2*math.pi, math.pi/25)
y = [math.cos(i) for i in x]

#autre methode pour obtenir x:
#for i in x:
#	y.append(math.cos(i))

plt.plot(x,y)
plt.xlabel('x : set of 100 real numbers i comprised between [-2 π, 2 π [')
plt.ylabel('y = cos(x)')
plt.title('Fonction : f(x) = cos(x) ; ∀ x ∈ [-2 π, 2 π [')
plt.show()
