import numpy as np

train_accuracies = [0.8202163371876166, 0.8166107173940073, 0.8142898586762817, 0.8057109702018318,
                    0.789464959177753, 0.8128393219777031, 0.8142484147706079, 0.8049649798997057, 0.8096895851465042,
                    0.8154917319408181]
test_accuracies = [0.811868059008785, 0.8180009945300846, 0.8267860102768109, 0.8037460633184154,
                   0.8045748383888612, 0.8092159787833582, 0.8148516492623902, 0.8113707939665175, 0.8075584286424664,
                   0.8219791148682247]
recalls = [0.6832917475089612, 0.7111466390030596, 0.722072528273056, 0.6777228178489301,
           0.6416673934491461, 0.6850675852707351, 0.7035694123647062, 0.6651485560064325, 0.6734149361783197,
           0.7072063013239483]
precisions = [0.7666070890908975, 0.7693654952440622, 0.7779740498319734, 0.7437043705628816,
              0.7531500509202992, 0.7638157354754145, 0.7608355861632365, 0.7780182232346241, 0.7621905500470836,
              0.7648992868354472]
f_1_scores = [0.7065723469648497, 0.7311489469885362, 0.7421811003268168, 0.6977214444224391,
              0.663785701492914, 0.7074705768981653, 0.7231625764281566, 0.6902043899436682, 0.6964490678468855,
              0.7274485472766063]

n_nodes_1 = [6, 6, 4, 5, 7, 12, 8, 6, 7, ]
n_nodes_2 = [2, 2, 2, 2, 3, 4, 2, 2, 3, ]

print('Average train accuracy: ' + str(np.mean(train_accuracies)))
print('Average test accuracy: ' + str(np.mean(test_accuracies)))
print('Average recall: ' + str(np.mean(recalls)))
print('Average precision: ' + str(np.mean(precisions)))
print('Average f_1 score: ' + str(np.mean(f_1_scores)))
print('Average connections 1. layer: ' + str(np.mean(n_nodes_1)))
print('Average connections 2. layer: ' + str(np.mean(n_nodes_2)))

print('Standard deviation of train accuracies: ' + str(np.std(train_accuracies)))
print('Standard deviation of test accuracies: ' + str(np.std(test_accuracies)))

print('Standard deviation of connections in first layer: ' + str(np.std(n_nodes_1)))
print('Standard deviation of connections in second layer: ' + str(np.std(n_nodes_2)))
