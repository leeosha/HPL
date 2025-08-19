
def hard_spotlight_attention(click_long_seq, target_item, pool_windows=[50,25,10,5]):
    # click_long_seq : [B, N, M, D] , B:Batch_Size, N: sequence len, M: feature number, D: feature dim
    # target_item : [B, 1, M, D]
  
    click_long_seq_len = click_long_seq.get_shape().as_list()[1]
    # [B, N, M, D]
    clk_target_item_features = tf.tile(target_item, [1,click_long_seq_len,1,1])
  
    # == hard matching == 
    # [B, N, M]
    binary_matrix = tf.cast(tf.reduce_all(tf.equal(click_long_seq, target_item), -1), tf.float32)
    # [B, N, M, 1]
    binary_matrix_4d = tf.expand_dims(binary_matrix, -1)
  
    # == max pooling == 
    max_flattens = []
    for i in pool_windows:
        max_pool_out = tf.nn.max_pool(binary_matrix_4d, 
                                                ksize=[1,i,1,1], strides=[1,i,1,1], padding='VALID')
        max_flatten = tf.layers.flatten(max_pool_out)
        max_flattens.append(max_flatten)
    max_out = tf.concat(max_flattens, axis=-1)
  
    # == sum pooling == 
    sum_flattens = []
    for i in pool_windows:
        kernel = tf.ones([i, 1, 1, 1], dtype=tf.float32) 
        conv_result = tf.nn.conv2d(
                input=binary_matrix_4d,
                filter=kernel,
                strides=[1, i, 1, 1],
                padding='VALID'
            )
        sum_flatten = tf.layers.flatten(conv_result)
        sum_flattens.append(sum_flatten)
    sum_out = tf.concat(sum_flattens, axis=-1)
    return tf.concat([max_out, sum_out], axis=-1)

def soft_spotlight_attention(click_long_seq, target_item, feature_dim, pool_windows=[50,25,10,5]):
    # click_long_seq : [B, N, M, D] , B:Batch_Size, N: sequence len, M: feature number, D: feature dim
    # target_item : [B, 1, M, D]
    click_long_seq_len = click_long_seq.get_shape().as_list()[1]
    # [B, N, M, D]
    target_item = tf.tile(target_item, [1,click_long_seq_len,1,1])
    # == cosine similarity ==
    # [B, N, M]
    cosine_sim =  tf.reduce_sum(tf.nn.l2_normalize(click_long_seq, axis=-1) *  tf.nn.l2_normalize(target_item, axis=-1),-1)
    # [B, N, M, 1]
    cosine_sim_4d = tf.expand_dims(cosine_sim, -1)
  
    # == max pooling== 
    max_flattens = []
    for i in pool_windows:
        max_pool_out = tf.nn.max_pool(cosine_sim_4d, 
                                                ksize=[1,i,1,1], strides=[1,i,1,1], padding='VALID')
        max_flatten = tf.layers.flatten(max_pool_out)
        max_flattens.append(max_flatten)
    max_out = tf.concat(max_flattens, axis=-1)
  
    # == sum pooling== 
    sum_flattens = []
    for i in pool_windows:
        kernel = tf.ones([i, 1, 1, 1], dtype=tf.float32) 
        conv_result = tf.nn.conv2d(
                input=cosine_sim_4d,
                filter=kernel,
                strides=[1, i, 1, 1],
                padding='VALID'
            )
        sum_flatten = tf.layers.flatten(conv_result)
        sum_flattens.append(sum_flatten)
    sum_out = tf.concat(sum_flattens, axis=-1)
  
    # == gate layer == 
    max_gate_thresholds = tf.get_variable(
                        shape=[max_out.get_shape().as_list()[-1]],
                        initializer=tf.glorot_normal_initializer(),
                        name='max_gate_thresholds')
    max_gate_out = tf.nn.relu(max_out - tf.nn.sigmoid(max_gate_thresholds))

    sum_gate_thresholds = tf.get_variable(
                        shape=[sum_out.get_shape().as_list()[-1]],
                        initializer=tf.glorot_normal_initializer(),
                        name='sum_gate_thresholds')
    sum_gate_out = tf.nn.relu(sum_out - tf.nn.sigmoid(sum_gate_thresholds))
    return tf.concat([max_gate_out, sum_gate_out], axis=-1)
