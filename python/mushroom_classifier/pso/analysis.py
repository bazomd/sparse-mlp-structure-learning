import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_accuracies = [0.9627692307692308, 0.9889230769230769, 0.9741538461538461,
                    0.9695384615384616, 0.9775384615384616, 0.978, 0.9856923076923076,
                    0.978, 0.9733846153846154, 0.9773846153846154]
test_accuracies = [0.9643076923076923, 0.9852307692307692, 0.9778461538461538,
                   0.9692307692307692, 0.9766153846153847, 0.9796923076923076, 0.9833846153846154,
                   0.9747692307692307, 0.9735384615384616, 0.9772307692307692]
recalls = [0.964337256784512, 0.9847133757961783, 0.9777306468716861,
           0.9692878805782033, 0.9766315617155954, 0.9796378442997493, 0.9820717131474104,
           0.9747713234447266, 0.9730681472467295, 0.9771543280631212]
precisions = [0.9641031831350327, 0.9861111111111112, 0.9782377919320595,
              0.9692793496999057, 0.9765752166519656, 0.9796844181459565, 0.9849833147942157,
              0.9747483012989532, 0.9738076157800278, 0.977267212617531]
f_1s = [0.964214332795694, 0.9851957801140607, 0.977835871789043,
        0.9692307575785117, 0.9766019073139975, 0.9796607585200087, 0.9832493989827089,
        0.9747594428097277, 0.9734075853174098, 0.9772074296387192]

n_nodes_1 = [3, 5, 4, 4, 6, 5, 3, 4, 4, 4]
n_nodes_2 = [2, 2, 3, 3, 2, 2, 2, 2, 2, 2]

print('Average train accuracy: ' + str(np.mean(train_accuracies)))
print('Average test accuracy: ' + str(np.mean(test_accuracies)))

print('Standard deviation of train accuracies: ' + str(np.std(train_accuracies)))
print('Standard deviation of test accuracies: ' + str(np.std(test_accuracies)))

print('Standard deviation of connections in first layer: ' + str(np.std(n_nodes_1)))
print('Standard deviation of connections in second layer: ' + str(np.std(n_nodes_2)))

# Feature analyses
plt.style.use('ggplot')
seaborn.set_theme(style="whitegrid")
dataset = pd.read_csv('../dataset/agaricus-lepiota.data')


g = seaborn.countplot(data=dataset, x='odor', hue='class')
g.figure.savefig("output.png")

