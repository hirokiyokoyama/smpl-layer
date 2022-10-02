import tensorflow as tf
from .constraints import NonNegUnitSum

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

def unique_edges(faces):
    edges1 = tf.gather(faces, [0,1], axis=1)
    edges2 = tf.gather(faces, [1,2], axis=1)
    edges3 = tf.gather(faces, [2,0], axis=1)
    edges = tf.concat([edges1, edges2, edges3], axis=0)
    shift = tf.where(edges[:,0:1] > edges[:,1:2], [[0, 32]], [[32, 0]])
    edges = tf.cast(edges, tf.int64)
    shift = tf.cast(shift, tf.int64)
    edges_packed = tf.bitwise.left_shift(edges, shift)
    edges_packed = tf.unique(tf.reduce_sum(edges_packed, axis=1)).y

    edges = tf.stack([
        tf.bitwise.bitwise_and(edges_packed, 0xffffffff),
        tf.bitwise.right_shift(edges_packed, 32)
    ], axis=1)
    return edges

class SMPL(tf.keras.layers.Layer):
    def __init__(self,
                 v_template,
                 faces,
                 child2parent,
                 n_betas = 10, 
                 simplify = False,
                 dtype = tf.float32,
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
                 posedirs_regularizer = 'l2',
                 edge_elasticity_loss_weight = 1.0):
        super().__init__(dtype=dtype)
        self.v_template = tf.constant(v_template, tf.float32)
        self.child2parent = tf.constant(child2parent, tf.int32)
        self.faces = tf.constant(faces, tf.int32)
        self.edges = unique_edges(self.faces)
        self.log_edge_lengths = tf.math.log(self.edge_length(self.v_template))
        n_vertices = int(self.v_template.shape[0])
        n_joints = len(child2parent)

        self.J_regressor = None
        self.weights_ = None
        self.shapedirs = None
        self.posedirs = None
        
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
        self.edge_elasticity_loss_weight = edge_elasticity_loss_weight

    def edge_length(self, vertices):
        edges = tf.gather(vertices, self.edges, axis=-2)
        length = tf.linalg.norm(edges[...,0,:] - edges[...,1,:], axis=-1)
        return length

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
        faces = tf.cast(data['f'], tf.int32)

        smpl = SMPL(
            data['v_template'], faces, child2parent,
            n_betas = n_betas,
            simplify = simplify,
            dtype = dtype)
        smpl.J_regressor.assign(data['J_regressor'].todense())
        smpl.weights_.assign(data['weights'])
        smpl.shapedirs.assign(data['shapedirs'])
        smpl.posedirs.assign(data['posedirs'])
        smpl.faces = tf.constant(data['f'], tf.int32)
        return smpl

    def vertices2joints(self, vertices):
        return tf.matmul(self.J_regressor, vertices)

    def call(self, inputs):
        betas = inputs['beta']
        pose = inputs['pose']
        N = tf.shape(betas)[0]

        v_template = self.v_template
        shapedirs = inputs.get('shapedirs', self.shapedirs)
        J_regressor = inputs.get('J_regressor', self.J_regressor)
        weights_ = inputs.get('weights', self.weights_)

        # add principal components to each template vertex
        #v_shaped = tf.tensordot(shapedirs, betas, axes=[[2], [1]]) + v_template
        # [N,B] [V,3,B] -> [N,V,3]
        v_shaped = tf.tensordot(betas, shapedirs, axes=[[-1], [-1]]) + v_template

        # estimate joint positions
        # [1,J,V] [N,V,3] -> [N,J,3]
        J = tf.matmul(J_regressor[tf.newaxis], v_shaped)

        # make batch of 3x3 rotation matrices
        pose_cube = tf.reshape(pose, [-1, 1, 3])
        R_cube_big = rodrigues(pose_cube)
        R_cube_big = tf.reshape(R_cube_big, [-1, 24, 3, 3])

        if self.simplify:
            v_posed = v_shaped
        else:
            # [V,3,(J-1)*9]
            posedirs = inputs.get('posedirs', self.posedirs)

            # posedirs represents distortion caused by joint rotation
            R_cube = R_cube_big[:,1:]
            lrotmin = tf.reshape(R_cube - tf.eye(3, dtype=self.dtype)[tf.newaxis,tf.newaxis], [-1,23*9])
            #v_posed = v_shaped + tf.tensordot(posedirs, lrotmin, axes=[[2], [0]])
            # [N,(J-1)*9] [V,3,(J-1)*9] -> [N,V,3]
            v_posed = v_shaped + tf.tensordot(lrotmin, posedirs, axes=[[-1], [-1]])

        # affine transform (4x4) of root joint
        # [N,3,3] [N,3,1] -> [N,3,4]
        # [N,3,4] [N,1,4] -> [N,4,4]
        bottom_row = tf.tile([[[0., 0., 0., 1.]]], [N,1,1])
        R = tf.concat([
            tf.concat([R_cube_big[:,0], J[:,0,:,tf.newaxis]], axis=-1),
            bottom_row], axis=-2)
        R = R[:,tf.newaxis]
        
        # stack affine transforms (Nx4x4) from global origin to joints
        for i in range(1, self.n_joints):
            j = self.child2parent[i]
            # affine transform (4x4) of joint i
            # [N,4,4] [N,4,4] -> [N,4,4]
            _R = tf.matmul(
                R[:,j],
                tf.concat([
                    # [N,3,4] [N,1,4] -> [N,4,4]
                    tf.concat([
                        # [N,3,3] [N,3,1] -> [N,3,4]
                        R_cube_big[:,i], (J[:,i,:] - J[:,j,:])[...,tf.newaxis]], axis=-1),
                    bottom_row], axis=-2)
            )
            # [N,J,3,3]
            R = tf.concat([R, _R[:,tf.newaxis]], axis=1)

        # make transforms relative to original poses
        # [N,J,4,4] [N,J,4,1] -> [N,J,4,1]
        RJ = tf.matmul(
            R,
            #tf.concat([J, tf.zeros([self.n_joints, 1], dtype=self.dtype)], axis=1)[:,:,tf.newaxis],
            tf.pad(J, [[0,0],[0,0],[0,1]])[...,tf.newaxis]
        )
        R = R - tf.pad(RJ, [[0,0], [0, 0], [0, 0], [3, 0]])

        # distribute transforms from joints to vertices
        #T = tf.tensordot(weights_, R, axes=[[1], [0]])
        #  [N,J,4,4] [V,J] -> [N,V,4,4]
        T = tf.einsum('njab,vj->nvab', R, weights_)

        # apply the weighted transforms for each vertex
        rest_shape_h = tf.pad(v_posed, [[0,0], [0,0], [0,1]], constant_values=1.)
        # [N,V,4,4] [N,V,4,1] -> [N,V,4,1]
        v = tf.matmul(T, rest_shape_h[...,tf.newaxis])
        # [N,V,3]
        v = v[...,:3,0]

        # penalize expansion/contraction of edges
        edge_lengths = tf.math.log(self.edge_length(v) + 1e-8)
        edge_elast = tf.square(edge_lengths - self.log_edge_lengths)
        edge_elast = tf.reduce_mean(edge_elast)
        self.add_loss(edge_elast * self.edge_elasticity_loss_weight)
        return v
