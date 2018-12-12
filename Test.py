# - read the input data:

import MnistLoader
training_data, validation_data, test_data = MnistLoader.load_data_wrapper()
training_data = list(training_data)

# ---------------------
# - network.py example:
from Network import Network, vectorized_result
from NetworkLoader import save, load



def predict(filename: str, y:int, net: Network):

    import cv2
    import numpy as np

    import os.path
    isExist = os.path.isfile(filename) 
    if(isExist):
        print("exist")
    else:
        print("not exist")
    
    # read image into matrix.
    m =  cv2.imread('./HandTestImages/0.png')
    
    # get image properties.
    h,w = np.shape(m)
    
    # iterate over the entire image.
    x = []
    for py in range(0,h):
        for px in range(0,w):
            print (m[py][px])
            x.append(px)
    x = np.array(x)
    y = y
    result = net.feedforward(x)
    print(np.argmax(result))


    

netPath = "E:\\ITMO University\\Интеллектуальные системы и технологии\\Lab5\Lab\\Models\\model_5epochs.json";

# net = Network([784, 30, 10])
# net.run(training_data, 5, 10, 3.0, test_data=test_data, monitor_evaluation_cost=True,
#             monitor_evaluation_accuracy=True,
#             monitor_training_cost=True,
#             monitor_training_accuracy=True)

# save(net, "E:\ITMO University\Интеллектуальные системы и технологии\Lab5\Lab\Models\model_5epochs.json")

net = load(netPath)

imgPath = "E:\\ITMO University\\Интеллектуальные системы и технологии\\Lab5\\Lab\\HandTestImages\\0.png"

predict(imgPath, 7,net)






# ----------------------
# - network2.py example:
# import network2


# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# #net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
#     monitor_evaluation_accuracy=True)

