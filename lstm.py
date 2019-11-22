# -*-coding:utf-8-*-#
import tensorflow as tf

class CLSTM_Cell(object):

    def __init__(self,config):
        num_nodes  = config.hidden_dim
        input_dim  = config.input_dim
        batch_size = config.batch_size
        output_dim = config.lables_num
        with tf.variable_scope("input", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as input_layer:
            self.ix, self.im, self.ib = self._generate_w_b(
                x_weights_size=[input_dim, num_nodes],
                m_weights_size=[num_nodes, num_nodes],
                biases_size=[1])
        with tf.variable_scope("memory", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as update_layer:
            self.cx, self.cm, self.cb = self._generate_w_b(
                x_weights_size=[input_dim, num_nodes],
                m_weights_size=[num_nodes, num_nodes],
                biases_size=[1])
        with tf.variable_scope("forget", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as forget_layer:
            self.fx, self.fm, self.fb = self._generate_w_b(
                x_weights_size=[input_dim, num_nodes],
                m_weights_size=[num_nodes, num_nodes],
                biases_size=[1])
        with tf.variable_scope("output", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as output_layer:
            self.ox, self.om, self.ob = self._generate_w_b(
                x_weights_size=[input_dim, num_nodes],
                m_weights_size=[num_nodes, num_nodes],
                biases_size=[1])

        self.w = tf.Variable(tf.truncated_normal([num_nodes, output_dim], -0.1, 0.1))
        self.b = tf.Variable(tf.zeros([output_dim]))

        self.saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        self.saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

    def _generate_w_b(self, x_weights_size, m_weights_size, biases_size):
        x_w = tf.get_variable("x_weights", x_weights_size)
        m_w = tf.get_variable("m_weigths", m_weights_size)
        b = tf.get_variable("biases", biases_size, initializer=tf.constant_initializer(0.0))
        return x_w, m_w, b

    def _run(self,index, input_sequence, output, state):
        input = input_sequence[:,index,:]
        forget_gate = tf.sigmoid(tf.matmul(input, self.fx) + tf.matmul(output, self.fm) + self.fb)
        input_gate = tf.sigmoid(tf.matmul(input, self.ix) + tf.matmul(output, self.im) + self.ib)
        update = tf.matmul(input, self.cx) + tf.matmul(output, self.cm) + self.cb
        state = state * forget_gate + tf.tanh(update) * input_gate
        output_gate = tf.sigmoid(tf.matmul(input, self.ox) + tf.matmul(output, self.om) + self.ob)
        return index+1, input_sequence, output_gate * tf.tanh(state), state

    def forward(self, input_sequence, labels, sequence_length):
        index = tf.constant(0)
        output = self.saved_output
        state = self.saved_state
        
        def cond(index,*args):
            return index < sequence_length
        
        index, _, output, state = tf.while_loop(cond, self._run, [index, input_sequence, output, state])
        
        with tf.control_dependencies([
                self.saved_output.assign(output),
                self.saved_state.assign(state)
            ]):
            # the length of outputs is batch_size
            logits = tf.nn.xw_plus_b(output, self.w, self.b)
            # the label should fix the size of ouputs
            
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits))
            train_prediction = tf.nn.softmax(logits)
        return logits, loss, train_prediction
