import random
import math
file_data='train.txt'
f=open(file_data,'w')
max=100
for i in range(max):
    a=complex((random.random()/2),(random.random()/2))
    b=complex((random.random()/2),(random.random()/2))
    c=a+b
    d=complex(1/(1+math.exp(-a.real)),1/(1+math.exp(-a.imag)))
    f.write('%s %s %s %s %s %s %s %s\n'%(a.real,a.imag,b.real,b.imag,c.real,c.imag,d.real,d.imag))
f.close()

file_data='test.txt'
f=open(file_data,'w')
max=10
for i in range(max):
    a=complex((random.random()/2),(random.random()/2))
    b=complex((random.random()/2),(random.random()/2))
    c=a+b
    d=complex(1/(1+math.exp(-a.real)),1/(1+math.exp(-a.imag)))
    f.write('%s %s %s %s %s %s %s %s\n'%(a.real,a.imag,b.real,b.imag,c.real,c.imag,d.real,d.imag))
f.close()