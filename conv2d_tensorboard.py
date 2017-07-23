import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('mnist_data',one_hot=True)


tf.reset_default_graph()

def conv2d(x,size_in,size_out,name):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fullyConnected(linput,size_in,size_out,name="fc"):
    with tf.name_scope(name):
        w_fc=tf.Variable(tf.truncated_normal([size_in,size_out], stddev=0.1),name="w")
        b_fc = tf.Variable(tf.constant(0.1, shape=[size_out]),name="b")
        y= tf.matmul(linput, w_fc) + b_fc
    return y

with tf.name_scope("X_Y"):
    x = tf.placeholder(tf.float32, shape=[None, 784],name="X_placeholder")
    y_ = tf.placeholder(tf.float32, shape=[None, 10],name="Y_placeholder")
    
x_image = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 3)



h_pool1 =conv2d(x_image, 1,32,"conv1")
h_pool2 = conv2d(h_pool1,32,64,"conv2")


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64],name="FLATTERN")



h_fc1 = tf.nn.relu(fullyConnected(h_pool2_flat,7 * 7 * 64,1024,name="fc1"),name="relu")


keep_prob = tf.placeholder(tf.float32,name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
y_out = fullyConnected(h_fc1_drop,1024,10,name="fc2")

with tf.name_scope("X-entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_out))
    tf.summary.scalar("cross_entropy", cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("Graphs")
writer.add_graph(sess.graph)

summ = tf.summary.merge_all()

for i in range(201):
    batch = mnist.train.next_batch(100)
    if i % 10 == 0:
        [train_accuracy,s] = sess.run([accuracy,summ],feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.5})
        writer.add_summary(s,i)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
