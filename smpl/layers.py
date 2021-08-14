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

class SMPL(tf.keras.layers.Layer):
    def __init__(self,
                 child2parent,
                 n_vertices = 6890,
                 n_betas = 10, 
                 simplify = False,
                 dtype = tf.float32):
        super().__init__(dtype=dtype)
        self.child2parent = tf.constant(child2parent, tf.int32)
        n_joints = len(child2parent)

        self.v_template = self.add_weight(
            'v_template',
            shape = [n_vertices, 3],
            trainable = True)
        self.J_regressor = self.add_weight(
            'J_regressor',
            shape = [n_joints, n_vertices],
            trainable = True)
        self.weights_ = self.add_weight(
            'weights',
            shape = [n_vertices, n_joints],
            trainable = True)
        self.shapedirs = self.add_weight(
            'shapedirs',
            shape = [n_vertices, 3, n_betas],
            trainable = True)
        if not simplify:
            self.posedirs = self.add_weight(
                'posedirs',
                shape = [n_vertices, 3, (n_joints-1)*9],
                trainable = True)
        self.n_vertices = n_vertices
        self.n_joints = n_joints
        self.n_betas = n_betas
        self.simplify = simplify

    @staticmethod
    def load_from_pkl(file, dtype=None):
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
            dtype = dtype)
        smpl.J_regressor.assign(data['J_regressor'].todense())
        smpl.weights_.assign(data['weights'])
        smpl.v_template.assign(data['v_template'])
        smpl.shapedirs.assign(data['shapedirs'])
        if not simplify:
            smpl.posedirs.assign(data['posedirs'])

        smpl.faces = tf.constant(data['f'], tf.int32)
        return smpl

    def vertices2joints(self, vertices):
        return tf.matmul(self.J_regressor, vertices)

    def call(self, inputs):
        betas, pose = inputs
        return self.generate(betas, pose)

    def generate(self, betas, pose):
        # add principal components to each template vertex
        v_shaped = tf.tensordot(self.shapedirs, betas, axes=[[2], [0]]) + self.v_template

        # estimate joint positions
        J = tf.matmul(self.J_regressor, v_shaped)

        # make batch of 3x3 rotation matrices
        pose_cube = tf.reshape(pose, [-1, 1, 3])
        R_cube_big = rodrigues(pose_cube)

        if self.simplify:
            v_posed = v_shaped
        else:
            # posedirs represents distortion caused by joint rotation
            R_cube = R_cube_big[1:]
            lrotmin = tf.reshape(R_cube - tf.eye(3, dtype=self.dtype)[tf.newaxis], [-1])
            v_posed = v_shaped + tf.tensordot(self.posedirs, lrotmin, axes=[[2], [0]])

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
            tf.concat([J, tf.zeros([24, 1], dtype=self.dtype)], axis=1)[:,:,tf.newaxis],
        )
        R = R - tf.pad(RJ, [[0, 0], [0, 0], [3, 0]])

        # distribute transforms from joints to vertices
        T = tf.tensordot(self.weights_, R, axes=[[1], [0]])

        # apply the weighted transforms for each vertex
        rest_shape_h = tf.pad(v_posed, [[0, 0], [0, 1]], constant_values=1.)
        v = tf.matmul(T, tf.reshape(rest_shape_h, [-1, 4, 1]))
        v = tf.reshape(v, [-1, 4])[:, :3]
        return v
