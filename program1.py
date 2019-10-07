# TTTGTAGTATTCTCTCATGGTAGTTTGTATTTCTGT
# TTTGTAGTATCCTCTCATGGTACTTTGTATTACTGT


file = open('data1.txt','r')
A = file.readline()
B = file.readline()

if len(A) == len(B):
	print('The two sequences have the same length.')
else:
	print("The two sequences doesn't have the same length.")

j = 0
for index, v in enumerate(A):
	if v != B[index]:
		j += 1
		print("This sequences differ at the position", index,"with the nucleotide", v, "for the A sequence and", B[index], "for the B sequence.")

a = ((len(A)-j)/len(A))*100
print("The percentage of identical nucleotides between these two sequences is :", a)

k = 0
for i in A:
	if i == 'G':
		k += 1
g = ((len(A)-k)/len(A))*100
print("The percentage of nucleotides G in sequence A is :", g)


k = 0
for i in B:
	if i == 'G':
		k += 1
g = ((len(A)-k)/len(A))*100
print("The percentage of nucleotides G in sequence B is :", g)


k = 0
for i in A:
	if i == 'C':
		k += 1
g = ((len(A)-k)/len(A))*100
print("The percentage of nucleotides C in sequence A is :", g)

k = 0
for i in B:
	if i == 'C':
		k += 1
g = ((len(A)-k)/len(A))*100
print("The percentage of nucleotides C in sequence B is :", g)

def compte(nuc, seq):
	k = 0
	for i in seq:
		if i == nuc:
			k += 1
	g = ((len(seq)-k)/len(seq))*100
	print("The percentage of nucleotides", nuc, "in sequence", seq,"is :", g)