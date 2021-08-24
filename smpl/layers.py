import tensorflow as tf

def rodrigues(r):
    # avoid division by zero
    noise = tf.random.normal(r.shape, 0, 1e-8, dtype=r.dtype)
    theta = tf.linalg.norm(r + noise, axis=[1, 2], keepdims=True)
    r_hat = r / theta

    cos = tf.math.cos(theta)
    z_stick = tf.zeros([theta.shape[0]], dtype=r.dtype)
    m = tf.stack(
        [z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
        -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick], axis=1)
    m = tf.reshape(m, [-1, 3, 3])
    i_cube = tf.expand_dims(tf.eye(3, dtype=r.dtype), axis=0) + tf.zeros(
        [theta.shape[0], 3, 3], dtype=r.dtype)
    A = tf.transpose(r_hat, [0, 2, 1])
    B = r_hat
    dot = tf.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + tf.math.sin(theta) * m
    return R

class NonNegUnitSum(tf.keras.constraints.Constraint):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def __call__(self, w):
        w = w * tf.cast(w >= 0., w.dtype)
        w = w / tf.reduce_sum(w, axis=self.axis, keepdims=True)
        return w

class SMPL(tf.keras.layers.Layer):
    def __init__(self,
                 child2parent,
                 n_vertices = 6890,
                 n_betas = 10, 
                 simplify = False,
                 dtype = tf.float32,
                 use_v_template = True,
                 v_template_initializer = 'random_normal',
                 v_template_constraint = None,
                 v_template_regularizer = None,
                 use_J_regressor = True,
                 J_regressor_initializer = 'random_normal',
                 J_regressor_constraint = NonNegUnitSum(axis=1),
                 J_regressor_regularizer = 'l1',
                 use_weights = True,
                 weights_initializer = 'random_normal',
                 weights_constraint = NonNegUnitSum(axis=1),
                 weights_regularizer = None,
                 use_shapedirs = True,
                 shapedirs_initializer = 'random_normal',
                 shapedirs_constraint = None,
                 shapedirs_regularizer = 'l2',
                 use_posedirs = True,
                 posedirs_initializer = 'random_normal',
                 posedirs_constraint = None,
                 posedirs_regularizer = 'l2'):
        super().__init__(dtype=dtype)
        self.child2parent = tf.constant(child2parent, tf.int32)
        n_joints = len(child2parent)

        if use_v_template:
            self.v_template = self.add_weight(
                'v_template',
                shape = [n_vertices, 3],
                initializer = v_template_initializer,
                constraint = v_template_constraint,
                regularizer = v_template_regularizer,
                trainable = True)
        if use_J_regressor:
            self.J_regressor = self.add_weight(
                'J_regressor',
                shape = [n_joints, n_vertices],
                initializer = J_regressor_initializer,
                constraint = J_regressor_constraint,
                regularizer = J_regressor_regularizer,
                trainable = True)
        if use_weights:
            self.weights_ = self.add_weight(
                'weights',
                shape = [n_vertices, n_joints],
                initializer = weights_initializer,
                constraint = weights_constraint,
                regularizer = weights_regularizer,
                trainable = True)
        if use_shapedirs:
            self.shapedirs = self.add_weight(
                'shapedirs',
                shape = [n_vertices, 3, n_betas],
                initializer = shapedirs_initializer,
                constraint = shapedirs_constraint,
                regularizer = shapedirs_regularizer,
                trainable = True)
        if not simplify:
            if use_posedirs:
                self.posedirs = self.add_weight(
                    'posedirs',
                    shape = [n_vertices, 3, (n_joints-1)*9],
                    initializer = posedirs_initializer,
                    constraint = posedirs_constraint,
                    regularizer = posedirs_regularizer,
                    trainable = True)
        self.n_vertices = n_vertices
        self.n_joints = n_joints
        self.n_betas = n_betas
        self.simplify = simplify

    @staticmethod
    def from_pkl(file, dtype=None):
        import pickle
        import numpy as np

        with open(file, 'rb') as f:
            data = pickle.load(f, encoding="latin1")

        if dtype is None:
            if data['v_template'].dtype == np.float64:
                dtype = tf.float64
            else:
                dtype = tf.float32
        simplify = 'posedirs' not in data
        n_vertices = data['v_template'].shape[0]
        n_joints = data['J_regressor'].shape[0]
        n_betas = data['shapedirs'].shape[2]

        kintree_table = tf.cast(data['kintree_table'], tf.int32)
        child2parent = tf.scatter_nd(kintree_table[1,:,tf.newaxis], kintree_table[0], [24])

        smpl = SMPL(
            child2parent,
            n_vertices = n_vertices,
            n_betas = n_betas,
            simplify = simplify,
            J_regressor_initializer = data['J_regressor'].todense(),
            weights_initializer = data['weights'],
            v_template_initializer = data['v_template'],
            shapedirs_initializer = data['shapedirs'],
            posedirs_initializer = data['posedirs'],
            dtype = dtype)
        smpl.faces = tf.constant(data['f'], tf.int32)
        return smpl

    def vertices2joints(self, vertices):
        return tf.matmul(self.J_regressor, vertices)

    def call(self, inputs):
        betas = inputs['beta']
        pose = inputs['pose']
        shapedirs = inputs.get('shapedirs') or self.shapedirs
        v_template = inputs.get('v_template') or self.v_template
        J_regressor = inputs.get('J_regressor') or self.J_regressor
        weights_ = inputs.get('weights') or self.weights_

        # add principal components to each template vertex
        v_shaped = tf.tensordot(shapedirs, betas, axes=[[2], [0]]) + v_template

        # estimate joint positions
        J = tf.matmul(J_regressor, v_shaped)

        # make batch of 3x3 rotation matrices
        pose_cube = tf.reshape(pose, [-1, 1, 3])
        R_cube_big = rodrigues(pose_cube)

        if self.simplify:
            v_posed = v_shaped
        else:
            posedirs = inputs.get('posedirs') or self.posedirs

            # posedirs represents distortion caused by joint rotation
            R_cube = R_cube_big[1:]
            lrotmin = tf.reshape(R_cube - tf.eye(3, dtype=self.dtype)[tf.newaxis], [-1])
            v_posed = v_shaped + tf.tensordot(posedirs, lrotmin, axes=[[2], [0]])

        # affine transform (4x4) of root joint
        R = tf.concat([
            tf.concat([R_cube_big[0], J[0, :, tf.newaxis]], axis=1),
            [[0., 0., 0., 1.]]], axis=0)
        R = R[tf.newaxis]
        
        # stack affine transforms (Nx4x4) from global origin to joints
        for i in range(1, self.n_joints):
            j = self.child2parent[i]
            # affine transform (4x4) of joint i
            _R = tf.matmul(
                R[j],
                tf.concat([
                    tf.concat([R_cube_big[i], (J[i, :] - J[j, :])[:, tf.newaxis]], axis=1),
                    [[0., 0., 0., 1.]]], axis=0)
            )
            R = tf.concat([R, _R[tf.newaxis]], axis=0)

        # make transforms relative to original poses
        RJ = tf.matmul(
            R,
            tf.concat([J, tf.zeros([self.n_joints, 1], dtype=self.dtype)], axis=1)[:,:,tf.newaxis],
        )
        R = R - tf.pad(RJ, [[0, 0], [0, 0], [3, 0]])

        # distribute transforms from joints to vertices
        T = tf.tensordot(weights_, R, axes=[[1], [0]])

        # apply the weighted transforms for each vertex
        rest_shape_h = tf.pad(v_posed, [[0, 0], [0, 1]], constant_values=1.)
        v = tf.matmul(T, tf.reshape(rest_shape_h, [-1, 4, 1]))
        v = tf.reshape(v, [-1, 4])[:, :3]
        return v
