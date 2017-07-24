# Importing the libraries
import cv2
import numpy as np
import pandas as pd
import tensorflow as  tf

# Importing data
data = pd.read_csv("fer2013.csv")

#Splitting the data into the Training set and Test set
train_data = data[data.Usage == "Training"]
public_data = data[data.Usage == "PrivateTest"]
private_data = data[data.Usage == "PublicTest"]

#preprocessing function 
def reshapeImage(dataset):
    x = dataset.pixels.str.split(" ").tolist()
    y = dataset.emotion.values
    images = []
    for image in np.array(x,dtype=float):
        images.append(image) #we can reshape also
    return np.array(images,dtype=float),y

#preprocessing
train_x , train_y = reshapeImage(train_data)
public_test_x,public_test_y=reshapeImage(public_data)
private_test_x,private_test_y=reshapeImage(private_data)


y_train_onehot = tf.one_hot(train_y,7)
y_public_test_onehot = tf.one_hot(public_test_y,7)
y_private_test_onehot = tf.one_hot(private_test_y,7)


with tf.Session() as session:
    y_train =  session.run(y_train_onehot)
    y_public = session.run(y_public_test_onehot)
    y_private = session.run(y_private_test_onehot)
    
tf.reset_default_graph()  
sess = tf.Session()    

#create place holders
with tf.name_scope("X_Y"):
    X = tf.placeholder(tf.float32, shape=[None, 2304],name="X_placeholder")
    Y = tf.placeholder(tf.float32, shape=[None, 7],name="Y_placeholder")

#conv2d Function
def conv2d(x,size_in,size_out,name="conv2d"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([3,3, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        activation1 = tf.nn.relu(conv1 + b)
        w2 = tf.Variable(tf.truncated_normal([3,3,size_out,size_out]),name="W2")
        b2 =tf.Variable(tf.constant(0.1,shape=[size_out]),name="B2")
        conv2 = tf.nn.conv2d(activation1, w2, strides=[1, 1, 1, 1], padding="SAME")
        activation2 = tf.nn.relu(conv2 + b2)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations1", activation1)
        tf.summary.histogram("activations2", activation2)
    return tf.nn.max_pool(activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
#fullyconnected layer function
def fullyConnected(linput,size_in,size_out,name="fc"):
    with tf.name_scope(name):
        w_fc=tf.Variable(tf.truncated_normal([size_in,size_out], stddev=0.1),name="w")
        b_fc = tf.Variable(tf.constant(0.1, shape=[size_out]),name="b")
        y= tf.matmul(linput, w_fc) + b_fc
        tf.summary.histogram("weightsFCC", w_fc)
        tf.summary.histogram("biasesFCC", b_fc)
    return y

#reshape x placeHolder
x_image = tf.reshape(X, [-1, 48, 48, 1])

h_pool1 =conv2d(x_image, 1,32,"conv1")
h_pool2 = conv2d(h_pool1,32,64,"conv2")
h_pool3 = conv2d(h_pool2,64,128,"conv3")

flattern = tf.reshape(h_pool3, [-1, 6*6*128],name="FLATTERN")

h_fc1 = tf.nn.relu(fullyConnected(flattern,6*6*128,512,name="fc1"),name="relu")

keep_prob = tf.placeholder(tf.float32,name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


y_ = fullyConnected(h_fc1_drop,512,7,name="fc2")


with tf.name_scope("X-entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_))
    tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
var = tf.argmax(y_, 1)
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("new")
writer.add_graph(sess.graph)
saver = tf.train.Saver()
summ = tf.summary.merge_all()
for _ in range(1000):
    for bid in range(int(train_x.shape[0]/300)):
        x1,y1=train_x[bid*100:(bid+1)*100],y_train[bid*100:(bid+1)*100]
        sess.run(train_step,feed_dict={X:x1, Y:y1,keep_prob: 0.5})
    if _%10==0:
        [a,s] =sess.run([accuracy, summ],feed_dict={X:x1, Y:y1,keep_prob: 0.5}) 
        writer.add_summary(s,_)
        print("steps %d accuracy %g" %(_,a))

save_path = saver.save(sess, "new/My_first_model_train")
print(sess.run(accuracy,feed_dict={X:public_test_x,Y:y_public,keep_prob: 1.0}))


