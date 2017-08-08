import tensorflow as tf

def add(sess):
	# tensorflow結構
	count = tf.Variable(0, name='counter')
	# print(state.name) 可以印出name
	one = tf.constant(1)
	newValue = tf.add(count, one)
	update = tf.assign(count, newValue)
	init = tf.global_variables_initializer()

	# Session開始
	sess.run(init)
	for _ in range(5):
		sess.run(update)
		print (sess.run(count))

def multiply(sess):
	# tensorflow結構
	matrix1 = tf.constant([[1, 2]])
	matrix2 = tf.constant([[3], [4]])
	product = tf.matmul(matrix1, matrix2)

	# Session開始
	result = sess.run(product)
	print(result)

def operation(sess):
	# tensorflow結構
	num1 = tf.constant(10.)
	num2 = tf.constant(20.)
	num3 = tf.constant(30.)
	num4 = tf.constant(40.)
	num5 = tf.constant(50.)
	result = num1 + num2 - num3 * num4 / num5
	init = tf.global_variables_initializer()

	# Session開始
	sess.run(init)
	print (sess.run(result))

def placeholder():
	# tensorflow結構
	input1 = tf.placeholder(tf.float32)
	input2 = tf.placeholder(tf.float32)
	output = tf.multiply(input1, input2)

	# Session開始
	with tf.Session() as sess:
		print(sess.run(output, feed_dict={input1: [8.], input2: [7.]}))

if __name__ == '__main__':
	# 加法
	sess = tf.Session()
	add(sess)
	sess.close()

	# 矩陣乘法
	sess = tf.Session()
	multiply(sess)
	sess.close()

	# 另一種四則運算
	sess = tf.Session()
	operation(sess)
	sess.close()

	# placeholder
	placeholder()