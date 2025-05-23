from aida.aida import *;

host = 'localhost'; dbname = 'bixi'; user = 'bixi'; passwd = 'bixi'; jobName = 'torchLinear'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);
def trainingLoop(dw):
    import numpy as np
    import tensorflow.compat.v1 as tf
    import logging
    tf.disable_v2_behavior()


    rng = np.random
    learningrate = 0.01
    epoch_size = 1000
    distance = dw.gmdata2017[:,2]
    duration = dw.gmdata2017[:,3]
    train_X = dw._extract_X(distance)
    train_Y = dw._extract_y(duration)
    n_samples = train_X.shape[0]

    X = tf.placeholder('float')
    Y = tf.placeholder('float')
    W = tf.Variable(rng.randn(),name="Weight")
    b = tf.Variable(rng.randn(),name="Bias")
    pred = tf.add(tf.multiply(X,W),b)
    error = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

    optimizer = tf.train.GradientDescentOptimizer(learningrate).minimize(error)
    init = tf.global_variables_initializer()
    display_step = 50

    logging.info("HERE")

    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Fit all training data
        for epoch in range(epoch_size):
            logging.info("HERE2")
            try:
                for (x, y) in zip(train_X, train_Y):
                    sess.run(optimizer, feed_dict={X: x, Y: y})
            except Exception as e:
                logging.info(f"EXCEPTION : {str(e)}")
                return

            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                c = sess.run(error, feed_dict={X: train_X, Y: train_Y})
                logging.info("Epoch:", '%04d' % (epoch + 1), "error=", "{:.9f}".format(c), \
                      "W=", sess.run(W), "b=", sess.run(b))

        logging.info("Optimization Finished!")
        training_error = sess.run(error, feed_dict={X: train_X, Y: train_Y})
        logging.info("Training error=", training_error, "W=", sess.run(W), "b=", sess.run(b), '\n')


dw._X(trainingLoop)


dw._close()
