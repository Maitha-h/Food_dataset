with open('C:\\Users\\Maith\\Documents\\Datasets\\Dataset_food-101\\meta\\classes.txt') as l:
    labels = l.read().splitlines()
    print(len(labels))

for i in labels:
    print(i)