import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#emotion
#0-Angry, 1-disgust, 2-fear, 3-happy , 4-sad , 5-surprise , 6 - neutral
data = pd.read_csv("train/fer2013.csv")
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
    y =  session.run(y_train)
    y_ = session.run(y_pub_test)
    y__ = session.run(y_private_test)
    
tf.reset_default_graph()  
sess = tf.Session()    

with tf.name_scope("X_Y_placeholder"):
    X =tf.placeholder(tf.float32,shape=[None,2304],name="X_placeholder")
    Y_ =tf.placeholder(tf.float32,shape=[None,7],name="Y_placeholder")
with tf.name_scope("W_B"):
    W= tf.Variable(tf.zeros([2304,7]),name="W")
    b= tf.Variable(tf.zeros([7]),name="b")
    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)
                         
with tf.name_scope("MatMul"):
    ytf= tf.matmul(X,W) + b
    
with tf.name_scope("X-ent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_,logits=ytf))
    tf.summary.scalar("cross_entropy", cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)




with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(ytf,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    

sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("/home/akshay/Desktop/1")
writer.add_graph(sess.graph)

    
summ = tf.summary.merge_all()


for _ in range(10):
    if _%5==0:
        [a,s] =sess.run([accuracy, summ],feed_dict={X:train_x, Y_:y}) 
        writer.add_summary(s,_)
    sess.run(train_step,feed_dict={X:train_x, Y_:y})

print(sess.run(accuracy,feed_dict={X:private_test_x, Y_:y__}))





