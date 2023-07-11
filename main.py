import os

unqualified_reference = os.listdir("./unqualified_reference_img")
if len(unqualified_reference) == 0:
    # run kmeans
    os.system('python kmeans.py')
else:
    # run svm
    os.system('python svm.py')