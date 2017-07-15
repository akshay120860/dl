import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
#emotion
#0-Angry, 1-disgust, 2-fear, 3-happy , 4-sad , 5-surprise , 6 - neutral
data = pd.read_csv("fer2013.csv")
train_raw = data[data.Usage == "Training"]
private_test_raw = data[data.Usage == "PrivateTest"]
public_test_raw = data[data.Usage == "PublicTest"]

def reshapeImage(dataset):
    x = dataset.pixels.str.split(" ").tolist()
    y = dataset.emotion.values
    images = []
    for image in np.array(x,dtype=float):
        images.append(image) #we can reshape also
    return np.array(images,dtype=float),y

train_x , train_y = reshapeImage(train_raw)
public_test_x,public_test_y=reshapeImage(public_test_raw)
private_test_x,private_test_y=reshapeImage(private_test_raw)

y_train = tf.one_hot(train_y,7)

y_pub_test = tf.one_hot(public_test_y,7)

y_private_test = tf.one_hot(private_test_y,7)

with tf.Session() as session:
    ytrain =  session.run(y_train)
    y1 = session.run(y_pub_test)
    y__ = session.run(y_private_test)

tf.reset_default_graph()  
sess = tf.Session()    


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
    x = tf.placeholder(tf.float32, shape=[None, 2304],name="X_placeholder")
    y_ = tf.placeholder(tf.float32, shape=[None, 7],name="Y_placeholder")
    
x_image = tf.reshape(x, [-1, 48, 48, 1])
#sess.run(x_image,feed_dict={x:})
tf.summary.image('input', x_image, 3)



h_pool1 =conv2d(x_image, 1,32,"conv1")
h_pool2 = conv2d(h_pool1,32,64,"conv2")


h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64],name="FLATTERN")



h_fc1 = tf.nn.relu(fullyConnected(h_pool2_flat,12 * 12 * 64,1024,name="fc1"),name="relu")


keep_prob = tf.placeholder(tf.float32,name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
y_out = fullyConnected(h_fc1_drop,1024,7,name="fc2")

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

writer = tf.summary.FileWriter("/home/akshay/Desktop/1")
writer.add_graph(sess.graph)

summ = tf.summary.merge_all()


for _ in range(120):
    t = (train_x.shape)[0]
    l=0
    while(l < t):
        
        if l + 100 < t:
            i,j = train_x[l:l+100,:],ytrain[l:l+100,:]
            l = l+100
        else:
            i,j = train_x[l:t,:],ytrain[l:t,:]
            l = t
        [g,h]=sess.run([accuracy,train_step],feed_dict={x:i, y_:j,keep_prob: 0.5})
        print l,g
    if _%2==0:
        [a,s] =sess.run([accuracy, summ],feed_dict={x:i, y_:j,keep_prob: 0.5}) 
        writer.add_summary(s,_)
        print("accuracy %g" %(a))

print(sess.run(accuracy,feed_dict={x:private_test_x, y_:y__,keep_prob: 1.0 }))
