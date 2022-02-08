import tensorflow as tf


def tf_lsh(plan_normals: tf.Tensor, points: tf.Tensor) -> tf.Tensor:
    """
    Compute the local sensitivity hashing for a list of points and a space partition. A space partition is given by
    a list of hyper plans that are origin centered and known by their normal vector
    :param plan_normals: the hyper plans normals. A tensor of shape (n,d) where n is the number of plans and
    d is the space dimension
    :param points: the points. A tensor of shape (k,d) where k is the number of points and d is the space dimension
    :return: a tensor of shape (k,) with type int64 containing the lsh for each point
    """
    (n, d) = plan_normals.shape
    (k, _) = points.shape
    plan_normals.shape.assert_has_rank(2)
    assert plan_normals.shape == (n, d)
    points.shape.assert_has_rank(2)
    assert points.shape[1] == d, \
        "points should have the same dimension as plans: %r vs %s " \
        % (d, points.shape[1])
    # stack points n times
    # shape=(k,n,d)
    n_stacked_points = tf.stack([points for i in range(n)], axis=1)
    assert n_stacked_points.shape == (k, n, d)
    # build dot product for each plan_to_point_vectors and normals to plans
    # shape=(k,n)
    #
    # first step: element wise multiplication of plan_to_point_vectors and plan_normals_tensor
    # shape=(k,n,d)
    eltWiseMult = tf.multiply(n_stacked_points, plan_normals)
    assert eltWiseMult.shape == (k, n, d)
    # second step: sum the multiplied values of the d axis
    dot_products = tf.reduce_sum(eltWiseMult, axis=2)
    assert dot_products.shape == (k, n)
    # for each dot product if > 0 yields 1 else 0
    position_indicators = tf.cast(dot_products > 0, tf.int64)
    # build powers of 2: one for each plane: shape=(1,n)
    powers_of_2 = tf.convert_to_tensor([[2 ** i for i in range(n)]], dtype=tf.int64)
    # result are dot products of position_indicators and powers_of_2
    result = tf.matmul(powers_of_2, tf.transpose(position_indicators))
    return tf.reshape(result, (k,))


def multi_tf_lsh(partitions: tf.Tensor, points: tf.Tensor) -> tf.Tensor:
    """
    computes local sensitivity hashing for a list of points, for multiple partitions
    :param partitions: a list of p partitions. A tensor of shape (p,n,d) where p is the number of space partitions,
    n is the number of plans per per space partition and d is the space dimension
    :param points: the points. A tensor of shape (k,d) where k is the number of points and d is the space dimension
    :return: a tensor of shape (p,k) containing for each partition the list of lsh-s for each point
    """
    partitions.shape.assert_has_rank(3)
    p, n, d = partitions.shape
    points.shape.assert_has_rank(2)
    k, d_p = points.shape
    assert d == d_p, "the space dimension is not the same (partitions has d=%s and points have d=%s )" % (d, d_p)
    result = tf.stack([tf_lsh(plan_normals, points) for plan_normals in partitions])
    return tf.reshape(result, (p, k))
