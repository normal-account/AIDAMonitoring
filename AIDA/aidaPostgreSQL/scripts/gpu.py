import GPUtil;

deviceID = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.5, maxMemory=0.5, attempts=1)
print(deviceID)
