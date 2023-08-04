from torch import nn
import torch
import math

import open3d as o3d
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_

from pykeops.torch import Vi, Vj


class DecreasingMLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, n_layers, in_feat_size, out_feat_size, final_layer_norm=False):
        super().__init__()

        size_step = (in_feat_size - out_feat_size) / (n_layers)
        layer_list = []

        for l in range(n_layers):
            layer_in_size = int(in_feat_size - l * size_step)
            layer_out_size = int(in_feat_size - (l + 1) * size_step)
            layer_list.append(nn.Linear(layer_in_size, layer_out_size))
            if l < (n_layers - 1):
                layer_list.append(nn.LeakyReLU())
            if (l < (n_layers - 1)) or final_layer_norm:
                layer_list.append(nn.LayerNorm(layer_out_size))
        self.layers = nn.Sequential(*layer_list)
        print("MLP", self.layers)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


class UnaryBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        # self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        # x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return "UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})".format(
            self.in_dim, self.out_dim, str(self.use_bn), str(not self.no_relu)
        )

class GridSampleConv(nn.Module):
    def __init__(
        self,
        in_fdim,
        out_fdim,
        subsampling_dist,
        kernel_radius,
        num_kernel_points=3,
        num_neighbors=32,
        preactivate=True,
        layernorm=False,
        deformable=False,
        relu=True,
        align_kp=False,
        kernel_debug_viz=False,
        grid_sampling_method="mean",
    ):
        """KPConv resnet Block with downsampling

        Args:
            in_fdim (int): feature dimension of input
            out_fdim (int): feature dimension of output
            subsampling_dist (float): resolution of the grid to subsample, no subsampling if <=0
            kernel_radius (float): radius of the convolutional kernel
            num_kernel_points (int, optional): num kernel points in each dimension of a grid(e.g 3=> 3^3 =27). Defaults to 3.
            num_neighbors (int, optional): Number neighbors for the convolution. Defaults to 32.
            preactivate (bool, optional): bool if preactivate. Don't do this if std(feature) is 0. Defaults to True.
            layernorm (bool, optional): bool if use layernorm. Defaults to True.
            deformable (bool, optional): bool if using deformable kpconv. Defaults to False.
            relu (bool, optional): bool if use relu. Defaults to True.
            kernel_debug_viz: visualize the input, query and kernel points for debugging and param tuning
        """
        super().__init__()
        self.in_fdim = in_fdim
        self.out_fdim = out_fdim
        conf_in_fdim = out_fdim if preactivate else in_fdim
        self.subsampling_dist = subsampling_dist
        self.kernel_radius = kernel_radius
        self.num_neighbors = num_neighbors
        self.grid_sampling_method = grid_sampling_method

        ### Preactivation ####
        self.relu = nn.LeakyReLU()
        if preactivate:
            pre_blocks = [nn.Linear(in_features=in_fdim, out_features=out_fdim)]
            if layernorm:
                pre_blocks.append(nn.LayerNorm(out_fdim))
            if relu:
                pre_blocks.append(self.relu)
            self.preactivation = nn.ModuleList(pre_blocks)
        else:
            self.preactivation = nn.ModuleList(
                [
                    nn.Identity(),
                ]
            )
        # KP Conv
        KP_extent = self.kernel_radius / (num_kernel_points - 1) * 1.5
        self.kp_conv = KPConv(
            kernel_size=num_kernel_points,
            p_dim=3,
            in_channels=conf_in_fdim,
            out_channels=out_fdim,
            KP_extent=KP_extent,
            radius=self.kernel_radius,
            align_kp=align_kp,
            kernel_debug_viz=kernel_debug_viz,
        )

        # print('kernel radius', self.kernel_radius)
        # Post linear
        post_layer = []
        if layernorm:
            post_layer.append(nn.LayerNorm(out_fdim))
        if relu:
            post_layer.append(self.relu)
        post_layer.append(nn.Linear(in_features=out_fdim, out_features=out_fdim))
        if layernorm:
            post_layer.append(nn.LayerNorm(out_fdim))
        self.post_layer = nn.ModuleList(post_layer)

        # Shortcut
        self.shortcut = nn.ModuleList(
            [
                nn.Identity(),
            ]
        )
        if in_fdim != out_fdim:
            sc_blocks = [nn.Linear(in_features=in_fdim, out_features=out_fdim)]
            if layernorm:
                sc_blocks.append(nn.LayerNorm(out_fdim))
            self.shortcut = nn.ModuleList(sc_blocks)

    def apply_module_list(self, module_list, features, mask):
        for block in module_list:
            if isinstance(block, nn.LayerNorm):
                features = block(features, mask)
            else:
                features = block(features)
        return features

    def forward(
        self,
        src_points: torch.Tensor,
        src_features: torch.Tensor,
        src_mask=None,
        query_points=None,
        query_features=None,
        query_mask=None,
        smpl_feats=None,
    ):
        """Computes a convolution for a subsampled subset of the input src_points

        Args:
            src_points (Tensor): [n x 3]
            features (Tensor): [n x in_fdim]
            smpl_feats (Tensor): [n x in_fdim] additional point features that get only grid sampled

        Returns:
            src_points: [m x 3], m <= n
            features: [m x out_fdim]
            sampled_smpl_feats: [m x out_fdim] additional point features that have been only grid sampled

        """
        if query_points == None:
            if self.subsampling_dist > 0:
                if self.grid_sampling_method == "mean":
                    sampling_return = meanGridSampling(
                        src_points,
                        features=src_features,
                        smpl_feats=smpl_feats,
                        resolution_meter=self.subsampling_dist,
                        mask=src_mask,
                        fill_value=0,
                    )
                elif self.grid_sampling_method == "random":
                    sampling_return = randomGridSampling(
                        src_points,
                        features=src_features,
                        smpl_feats=smpl_feats,
                        resolution_meter=self.subsampling_dist,
                        mask=src_mask,
                        fill_value=0,
                    )
                else:
                    raise ValueError("{} grid sampling method is not implemented.")
                query_points, query_features, query_mask = sampling_return[:3]
            else:
                query_points = src_points
                query_features = src_features
                query_mask = src_mask
        neighbors_index = masked_knn_keops(
            query_points,
            src_points,
            k=self.num_neighbors,
            q_mask=query_mask,
            s_mask=src_mask,
        )

        out_features = self.apply_module_list(
            self.preactivation, src_features, src_mask
        )
        out_features = self.kp_conv.forward(
            q_pts=query_points,
            s_pts=src_points,
            neighb_inds=neighbors_index,
            x=out_features,
        )

        out_features = self.apply_module_list(self.post_layer, out_features, query_mask)
        out_features = self.relu(
            self.apply_module_list(self.shortcut, query_features, query_mask)
            + out_features
        )
        # if out_features.isnan().all():
        #     import ipdb

        #     ipdb.set_trace()
        if smpl_feats == None:
            return query_points, out_features, query_mask
        else:
            # return also sampled feats
            return query_points, out_features, query_mask, sampling_return[-1]


class UpsampleBlock(nn.Module):
    def __init__(self):
        """Nearest Neighbor upsampling"""
        super().__init__()

    def forward(
        self,
        query_points: torch.Tensor,
        target_points: torch.Tensor,
        target_features: torch.Tensor,
        q_mask=None,
        t_mask=None,
    ):
        """Gets for each query point the feature of the nearest target point

        Args:
            query_points (torch.Tensor): [n x 3]
            target_points (torch.Tensor): [m x 3]
            target_features (torch.Tensor): [m x f_dim]

        Returns:
            query_points (torch.Tensor): [n x 3]
            query_features (torch.Tensor): [n x f_dim]
        """
        idx = masked_knn_keops(
            query_points, target_points, q_mask=q_mask, s_mask=t_mask, k=1
        )  # get nearest neighbor
        target_shape = list(target_features.shape)
        target_shape[-2] = idx.shape[-2]
        return query_points, torch.gather(target_features, -2, idx.expand(target_shape))


##############################################################
# KKPConv
##############################################################


def vector_gather(vectors: torch.Tensor, indices: torch.Tensor):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[B, N1, D]
        indices: Tensor[B, N2, K]
    Returns:
        Tensor[B,N2, K, D]
    """
    # out = vectors.transpose(0, 1)[indices, 0]
    # out = torch.gather(vectors,dim=-2,index= indices)

    # Out
    shape = list(indices.shape) + [vectors.shape[-1]]
    out = torch.zeros(shape, device=vectors.device)

    # src
    vectors = vectors.unsqueeze(-2)
    shape = list(vectors.shape)
    shape[-2] = indices.shape[-1]
    vectors = vectors.expand(shape)

    # Do the magic
    indices = indices.unsqueeze(-1).expand_as(out)
    out = torch.gather(vectors, dim=-3, index=indices)
    return out


def masked_knn_keops(query, target, k, q_mask=None, s_mask=None, metric="euclidean"):
    if not query.is_contiguous():  # TODO find out why query is not contiguous
        query = query.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    if q_mask is not None:
        query[~q_mask.expand(query.shape)] = 1e6
    if s_mask is not None:
        target[~s_mask.expand(target.shape)] = 1e6
        # print(s_mask.sum(-2))
        assert (s_mask.sum(-2) >= k).all()
    # Encoding as KeOps LazyTensors:
    D = query.shape[-1]
    X_i = Vi(0, D)  # Purely symbolic "i" variable, without any data array
    X_j = Vj(1, D)  # Purely symbolic "j" variable, without any data array

    # Symbolic distance matrix:
    if metric == "euclidean":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif metric == "manhattan":
        D_ij = (X_i - X_j).abs().sum(-1)
    elif metric == "angular":
        D_ij = -(X_i | X_j)
    elif metric == "hyperbolic":
        D_ij = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])
    else:
        raise NotImplementedError(f"The '{metric}' distance is not supported.")

    # K-NN query operator:
    KNN_fun = D_ij.argKmin(k, dim=1)

    # N.B.: The "training" time here should be negligible.
    # elapsed = timer() - start

    # target = tensor(target)
    # start = timer()
    # Actual K-NN query:
    indices = KNN_fun(query, target)

    # elapsed = timer() - start

    return indices

class KPConv(nn.Module):
    def __init__(
        self,
        kernel_size,
        in_channels,
        out_channels,
        radius,
        KP_extent=None,
        p_dim=3,
        radial=False,
        align_kp=False,
        kernel_debug_viz=False,
    ):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param radius: radius used for kernel point init.
        :param KP_extent: influence radius of each kernel point. (float), default: None
        :param p_dim: dimension of the point space. Default: 3
        :param radial: bool if direction independend convolution
        :param align_kp: aligns the kernel points along the main directions of the local neighborhood
        :param kernel_debug_viz: visualize the input, query and kernel points for debugging and param tuning
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.radial = radial
        self.p_dim = 1 if radial else p_dim  # 1D for radial convolution

        self.K = kernel_size**self.p_dim
        self.num_kernels = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = (
            radius / (kernel_size - 1) * self.p_dim**0.5
            if KP_extent is None
            else KP_extent
        )
        self.align_kp = align_kp
        self.kernel_debug_viz = kernel_debug_viz

        # Initialize weights
        self.weights = Parameter(
            torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
            requires_grad=True,
        )

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    @staticmethod
    def rotate_kp(kp, neighbors: torch.Tensor, radius):
        with torch.no_grad():
            # m = neighbors.mean(-2, keepdim=True) > 0
            # print("mean", m)
            within = neighbors.norm(dim=-1, keepdim=True) < radius
            n = neighbors * within
            cross_cov = n.transpose(-2, -1) @ n
            _, _, V = torch.linalg.svd(cross_cov)
            # V = torch.where(m.transpose(-2,-1).expand_as(V),V,-V)
            V.unsqueeze_(-3)
            return kp @ V

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a grid
        :return: the tensor of kernel points
        """

        K_points_numpy = getKernelPoints(self.radius, self.num_kernels, dim=self.p_dim)
        return Parameter(
            torch.tensor(K_points_numpy, dtype=torch.float32), requires_grad=False
        )

    def visualize_pcl_and_kernels(self, q_pts, s_pts, neighbors):
        print(
            "Shapes q:{} s:{} neigh:{}".format(
                q_pts.shape, s_pts.shape, neighbors.shape
            )
        )
        src_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                s_pts[0][(s_pts[0] != 1e6).all(-1)].detach().cpu().numpy()
            )
        )
        src_cloud.paint_uniform_color(np.array((1.0, 0, 0)))

        neighb_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                neighbors[0][::100][(neighbors[0][::100] != 1e6).all(-1)]
                .detach()
                .cpu()
                .numpy()
            )
        )
        neighb_cloud.paint_uniform_color(np.array((1.0, 0, 1.0)))
        neighb_cloud.translate(np.array((-0.0002, -0.0002, -0.0002)))

        query_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                q_pts[0][(q_pts[0] != 1e6).all(-1)].detach().cpu().numpy()
            )
        )
        query_cloud.paint_uniform_color(np.array((0, 1.0, 0)))
        query_cloud.translate(np.array((0.0002, 0.0002, 0.0002)))
        kernel_points = q_pts[0][(q_pts[0] != 1e6).all(-1)][::100].unsqueeze(
            -2
        ) + self.kernel_points.unsqueeze(-3)
        shape = kernel_points.shape
        kernel_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                kernel_points.reshape(shape[0] * shape[1], shape[2])
                .detach()
                .cpu()
                .numpy()
            )
        )
        kernel_cloud.paint_uniform_color(np.array((0, 0, 1.0)))
        o3d.visualization.draw_geometries([query_cloud, kernel_cloud, neighb_cloud])

    def forward(self, q_pts, s_pts, neighb_inds, x):
        # Add a fake point/feature in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[..., :1, :]) + 1e6), -2)
        x = torch.cat((x, torch.zeros_like(x[..., :1, :])), -2)

        # Get neighbor points and features [n_points, n_neighbors, dim/ in_fdim]
        if len(neighb_inds.shape) < 3:
            neighbors = s_pts[neighb_inds, :]
            neighb_x = gather(x, neighb_inds)
        else:
            neighbors = vector_gather(s_pts, neighb_inds)
            neighb_x = vector_gather(x, neighb_inds)

        if self.kernel_debug_viz:
            self.visualize_pcl_and_kernels(q_pts, s_pts, neighbors)

        # Center every neighborhood [n_points, n_neighbors, dim]
        neighbors = neighbors - q_pts.unsqueeze(-2)

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        if self.radial:
            neighbors = torch.sqrt(torch.sum(neighbors**2, -1, keepdim=True))
        kernel_points = (
            self.rotate_kp(self.kernel_points, neighbors, self.radius)
            if self.align_kp
            else self.kernel_points
        )
        neighbors.unsqueeze_(-2)
        # kernel_points.transpose_(-4, -3)
        # print(neighbors.shape, kernel_points.shape)
        differences = neighbors - kernel_points
        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences**2, dim=-1)
        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        all_weights = torch.clamp(
            1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0
        )

        fx = torch.einsum(
            "...nkl,...nki,...lio->...no", all_weights, neighb_x, self.weights
        )
        return fx

    def __repr__(self):
        return "KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})".format(
            self.radius, self.in_channels, self.out_channels
        )
        
        
def meanGridSampling(
    pcd: torch.Tensor,
    resolution_meter,
    scale=1.0,
    features=None,
    smpl_feats=None,
    mask=None,
    fill_value=0,
):
    """Computes the mean over all points in the grid cells

    Args:
        pcd (torch.Tensor): [...,N,3] point coordinates
        resolution_meter ([type]): grid resolution in meters
        scale (float, optional): Defaults to 1.0. Scale to convert resolution_meter to grid_resolution: resolution = resolution_meter/scale
        features (torch.Tensor): [...,N,D] point features
        smpl_feats (torch.Tensor): [...,N,D] additional point features to be grid sampled
    Returns:
        grid_coords (torch.Tensor): [...,N,3] grid coordinates
        grid_features (torch.Tensor): [...,N,D] grid features
        grid_smpl_feats (torch.Tensor): [...,N,D] additional point features to have been grid sampled
        mask (torch.Tensor): [...,N,1] valid point mask (True: valid, False: not valid)
    """
    resolution = resolution_meter / scale
    if len(pcd.shape) < 3:
        pcd = pcd.unsqueeze(0)
    if len(features.shape) < 3:
        features = features.unsqueeze(0)
    B = pcd.shape[0]

    grid_coords = torch.zeros_like(pcd, device=pcd.device)
    grid_features = torch.zeros_like(features, device=pcd.device)
    if smpl_feats != None:
        if len(smpl_feats.shape) < 3:
            smpl_feats = smpl_feats.unsqueeze(0)
        grid_smpl_feats = torch.zeros_like(smpl_feats, device=pcd.device)
    out_mask = torch.full_like(pcd[..., :1], False, dtype=bool, device=pcd.device)

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = float("inf")
    grid = torch.floor((pcd - pcd.min(dim=-2, keepdim=True)[0]) / resolution).double()

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = fill_value

    # v_size = math.ceil(1 / resolution)
    # grid_idx = grid[..., 0] + grid[..., 1] * v_size + grid[..., 2] * v_size * v_size
    if mask is not None:
       grid_size = grid[mask.squeeze(-1)].max().detach() + 1
    else:
       grid_size = grid.max().detach() + 1
    grid_idx = (
       grid[..., 0] + grid[..., 1] * grid_size + grid[..., 2] * grid_size * grid_size
   )

    max_nr = []
    for i in range(B):
        unique, indices, counts = torch.unique(
            grid_idx[i], return_inverse=True, dim=None, return_counts=True
        )
        indices.unsqueeze_(-1)

        nr_cells = len(counts)
        if unique[-1].isinf():
            counts = counts[:-1]
            nr_cells -= 1
        max_nr.append(nr_cells)

        grid_coords[i].scatter_add_(-2, indices.expand(pcd[i].shape), pcd[i])
        grid_coords[i, :nr_cells, :] /= counts.unsqueeze(-1)

        grid_features[i].scatter_add_(
            -2, indices.expand(features[i].shape), features[i]
        )
        grid_features[i, :nr_cells, :] /= counts.unsqueeze(-1)
        if smpl_feats != None:
            grid_smpl_feats[i].scatter_add_(
                -2, indices.expand(smpl_feats[i].shape), smpl_feats[i]
            )
            grid_smpl_feats[i, :nr_cells, :] /= counts.unsqueeze(-1)
        out_mask[i, :nr_cells, :] = True

        if fill_value != 0:
            grid_coords[i, nr_cells:] = fill_value

    max_nr = max(max_nr)
    grid_coords = grid_coords[..., :max_nr, :]
    grid_features = grid_features[..., :max_nr, :]
    out_mask = out_mask[..., :max_nr, :]
    if smpl_feats != None:
        grid_smpl_feats = grid_smpl_feats[..., :max_nr, :]
        return grid_coords, grid_features, out_mask, grid_smpl_feats
    else:
        return grid_coords, grid_features, out_mask


def randomGridSampling(
    pcd: torch.Tensor,
    resolution_meter,
    scale=1.0,
    features=None,
    smpl_feats=None,
    mask=None,
    fill_value=0,
):
    """Computes a rondom point over all points in the grid cells

    Args:
        pcd (torch.Tensor): [...,N,3] point coordinates
        resolution_meter ([type]): grid resolution in meters
        scale (float, optional): Defaults to 1.0. Scale to convert resolution_meter to grid_resolution: resolution = resolution_meter/scale
        features (torch.Tensor): [...,N,D] point features
        smpl_feats (torch.Tensor): [...,N,D] additional point features to be grid sampled
    Returns:
        grid_coords (torch.Tensor): [...,N,3] grid coordinates
        grid_features (torch.Tensor): [...,N,D] grid features
        grid_smpl_feats (torch.Tensor): [...,N,D] additional point features to have been grid sampled
        mask (torch.Tensor): [...,N,1] valid point mask (True: valid, False: not valid)
    """
    resolution = resolution_meter / scale
    if len(pcd.shape) < 3:
        pcd = pcd.unsqueeze(0)
    if len(features.shape) < 3:
        features = features.unsqueeze(0)
    B = pcd.shape[0]

    grid_coords = torch.zeros_like(pcd, device=pcd.device)
    grid_features = torch.zeros_like(features, device=pcd.device)
    if smpl_feats != None:
        if len(smpl_feats.shape) < 3:
            smpl_feats = smpl_feats.unsqueeze(0)
        grid_smpl_feats = torch.zeros_like(smpl_feats, device=pcd.device)
    out_mask = torch.full_like(pcd[..., :1], False, dtype=bool, device=pcd.device)

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = float("inf")
    grid = torch.floor((pcd - pcd.min(dim=-2, keepdim=True)[0]) / resolution)

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = fill_value

    if mask is not None:
        grid_size = grid[mask.squeeze(-1)].max().detach()
    else:
        grid_size = grid.max().detach()
    grid_idx = (
        grid[..., 0] + grid[..., 1] * grid_size + grid[..., 2] * grid_size * grid_size
    )

    max_nr = []
    for i in range(B):
        unique, indices, counts = torch.unique(
            grid_idx[i], return_inverse=True, dim=None, return_counts=True
        )

        nr_cells = len(counts)
        if unique[-1].isinf():
            counts = counts[:-1]
            nr_cells -= 1
        max_nr.append(nr_cells)
        indices.detach_()
        grid_point_ids = torch.full(
            pcd.shape[-2:-1], -1, device=pcd.device, dtype=torch.long
        )
        grid_point_ids.scatter_(
            -1, indices, torch.arange(len(indices), device=grid_point_ids.device)
        )
        grid_point_ids = grid_point_ids[:nr_cells].detach()

        grid_coords[i, :nr_cells, :] = pcd[i, grid_point_ids]

        grid_features[i, :nr_cells, :] = features[i, grid_point_ids]
        if smpl_feats != None:
            grid_smpl_feats[i, :nr_cells, :] = smpl_feats[i][grid_point_ids[:nr_cells]]

        out_mask[i, :nr_cells, :] = True

        if fill_value != 0:
            grid_coords[i, nr_cells:] = fill_value

    max_nr = max(max_nr)
    grid_coords = grid_coords[..., :max_nr, :]
    grid_features = grid_features[..., :max_nr, :]
    out_mask = out_mask[..., :max_nr, :]
    if smpl_feats != None:
        grid_smpl_feats = grid_smpl_feats[..., :max_nr, :]
        return grid_coords, grid_features, out_mask, grid_smpl_feats
    else:
        return grid_coords, grid_features, out_mask

def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')

# Grid Based Kernel Point initialization


def getKernelPoints(radius, num_points=3,dim=3):
    """[summary]

    Args:
        radius (float): radius
        num_points (int, optional): Number of kernel points per dimension. Defaults to 3.

    Returns:
        [type]: returns num_points^3 kernel points 
    """
    xyz = np.linspace(-1, 1, num_points)
    if dim ==1:
        return xyz[:,None]*radius

    points = np.meshgrid(*(dim*[xyz]))
    points = [p.flatten() for p in points]
    points = np.vstack(points).T
    points /= dim**(0.5) # Normalizes to stay in unit sphere
    return points*radius
