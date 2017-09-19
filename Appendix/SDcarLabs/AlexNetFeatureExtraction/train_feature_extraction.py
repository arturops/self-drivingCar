import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

num_classes = 43
epochs = 10
batch_size = 128
learn_rate = 1e-3

# TODO: Load traffic signs data.
file = open('train.p','rb')
data = pickle.load(file)

# TODO: Split data into training and validation sets.
x_train, x_validation, y_train, y_validation = train_test_split(data['features'],
																data['labels'],
																test_size=0.2)

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32,shape=(None,32,32,3),name='inputs')
labels = tf.placeholder(tf.int64,shape=(None),name='labels')

features_resized = tf.image.resize_images(features,(227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(features_resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
fc8Shape = (fc7.get_shape().as_list()[-1],num_classes)
fc8W = tf.Variable(tf.truncated_normal(fc8Shape,stddev=1e-2),name='fc8Weights')
fc8b = tf.Variable(tf.zeros(num_classes),name='fc8bias')
logits = tf.matmul(fc7,fc8W) + fc8b


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learn_rate)
train_step = optimizer.minimize(loss, var_list=(fc8W,fc8b))

predictions = tf.argmax(logits,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions,labels),tf.float32))

init_var = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.

def evaluate(xfeatures, ylabels, sess):
	total_accuracy = 0
	total_loss = 0

	for offset in range(0,ylabels.shape[0],batch_size):
		end = offset+batch_size
		batch_x, batch_y = xfeatures[offset:end], ylabels[offset:end]
		loss_batch, acc = sess.run([loss, accuracy],feed_dict={features: batch_x, labels: batch_y})
		
		# weighted average
		total_accuracy+=(batch_x.shape[0] * acc)
		total_loss+=(batch_x.shape[0] * loss_batch)
	
	total_accuracy/=xfeatures.shape[0]
	total_loss/=ylabels.shape[0]

	return total_accuracy

# Start the training session here

print('Train Dataset total size: {}'.format(y_train.shape[0]))
print('Validation Dataset total size: {}'.format(y_validation.shape[0]))
train_limit = y_train.shape[0]
val_limit = y_validation.shape[0]
x_train, y_train = x_train[0:train_limit], y_train[0:train_limit]
print('Dataset current samples used: {}'.format(y_train.shape[0]))

with tf.Session() as sess:
	sess.run(init_var)
	total_time = 0;
	print(' Start training on {} samples ...'.format(y_train.shape[0]))
	for epoch in range(epochs):

		x, y = shuffle(x_train, y_train)
		t_start = time.time() 

		batch_end = y.shape[0]
		for offset in range(0,batch_end,batch_size):
			end = offset+batch_size
			batch_x, batch_y = x[offset:end], y[offset:end]

			sess.run(train_step, feed_dict={features:batch_x, labels:batch_y})

		# check accuracy after every epoch
		val_accuracy = evaluate(x_validation[0:val_limit], y_validation[0:val_limit], sess)
		print('Epoch {}/{} ... Validation Accuracy {:.4f} ... Ellapsed time {:.4f} ...'.\
				format(epoch+1,epochs,val_accuracy,time.time()-t_start))
		total_time+=time.time()-t_start
	print('Total Training time {:.4f} s'.format(total_time))



