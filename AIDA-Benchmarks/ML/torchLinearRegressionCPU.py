from aida.aida import *;
host = 'localhost'; dbname = 'bixi'; user = 'bixi'; passwd = 'bixi'; jobName = 'torchLinear'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);


def trainingLoop(dw,input_size, output_size):
    import time
    import logging
    import sys

    import numpy as np
    nn = sys.modules["torch.nn.modules"]
    torch = sys.modules["torch"]
    datasets = sys.modules["sklearn.datasets"]
    start_time = time.time()
    learningrate = 0.0000001
    epoch_size = 100
    model = nn.Linear(input_size,output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningrate)
    distance = dw.gmdata2017[:,2]
    duration = dw.gmdata2017[:,3]
    X = dw._extract_X(distance)
    y = dw._extract_y(duration)
    X = np.copy(X)
    y = np.copy(y)
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    y = y.view(y.shape[0],1)
    for epoch in range(epoch_size):
        y_predicted = model(X)
        loss = criterion(y_predicted, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(model.weight)
    dw.linearModel = model
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info("execution time is "+str(execution_time))
    return(model.weight)


weight = dw._X(trainingLoop,1,1)
print(weight)
dw._tables()
dw._close()