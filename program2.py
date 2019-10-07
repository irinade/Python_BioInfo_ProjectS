ARN = {'A': 'U', 'T': 'A', 'C': 'G', 'G': 'C'}

file = open('data1.txt', 'r')
A = file.readline()
B = file.readline()
ARN_A = ""
A = A.rstrip("\n\r")
codons = []
codons2 = []
l1 = ""
l2 = ""
prot = {}

for i in A:
    for key, val in ARN.items():
        if i == key:
            ARN_A += val

print("ADN : ", A)
print("ARN : ", ARN_A)


for i in range(0, len(A), 3):
    codons.append(A[i:i + 3])
print("codons", codons)


file = open('data2.txt', 'r')
lines = file.readlines()

for i in lines:
    k = i.strip()
    l1 = k[0:3]
    l2 = k[4:]
    prot[l1] = l2
print("prot", prot)

for i in range(0, len(ARN_A), 3):
    codons2.append(ARN_A[i:i + 3])
print("Codon de l'ARN : ", codons2)

seq = []
for codon in codons2:
    for i in prot:
        if i == codon:
            seq += prot.get(i)
seq = "-".join(seq)
print("The protein sequence is : ", seq)

f = open('seqP.txt', "w+")
f.write(seq)
f.close()
