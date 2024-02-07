



import os
for dirname, _, filenames in os.walk('/kaggle/input/gtsrb-german-traffic-sign/Train.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))