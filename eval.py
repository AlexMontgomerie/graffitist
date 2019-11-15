import os,csv,sys
import numpy as np
 
# output path
output_path=sys.argv[1]

# read lines and accumulate
top1 = 0.0
top5 = 0.0

size =0.0

def get_top5(arr):
    tmp = np.array([float(x) for x in arr])
    return tmp.argsort()[-5:][::-1]

# iterate over files
for filename in os.listdir(output_path):
    with open(os.path.join(output_path,filename),'r') as f:
        reader = csv.reader(f,delimiter=",")
        for row in reader:
            # get class
            class_true  = int(row[1])
            pred = get_top5(row[2:])
            if pred[0] == class_true:
                top1 += 1
            if class_true in pred:
                top5 += 1
            size += 1

print("top1: ",top1/size, ", top5: ",top5/size)
