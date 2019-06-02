#tensorflow 불러오기
import tensorflow as tf
#하루 공부 시간
x_data=[1,2,3,4,5,6,7]
#물리 성적
y_data=[31,42,48,55,63,74,86]

#가중치
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
#편향
b=tf.Variable(tf.random_uniform([1],-1.0,1.0))

X=tf.placeholder(tf.float32, name="X")
Y=tf.placeholder(tf.float32, name="Y")

#가설식
H=W*X+b

#cost함수
cost=tf.reduce_mean(tf.square(H-Y))
#경사하강법
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

#학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        _, cost_val=sess.run([train, cost], feed_dict={X:x_data, Y:y_data})

        print(step, cost_val, sess.run(W), sess.run(b))
    #예측
    how=input('몇시간 공부하셨나요? ')
    print('당신의 성적은', sess.run(H, feed_dict={X:how}), '점으로 예상됩니다.')