import torch
import torch.nn as nn

from .blocks import DecreasingMLP, UpsampleBlock, GridSampleConv


class Decoder(nn.Module):
    def __init__(
        self,
        num_enc_layers,
        output_size,
        in_dims,
        out_dim,
        layer_kernel_radii,
        grid_sampling_method,
        enable_dropout=False,
        pos_enc=False,
        kernel_debug_viz=False,
        sampling_algo="legacy",
    ):
        super().__init__()

        self.enable_dropout = enable_dropout
        self.num_enc_layers = num_enc_layers
        self.pos_enc = pos_enc
        if enable_dropout:
            self.dropout = nn.Dropout(p=0.2)
        self.lin = torch.nn.ModuleList()
        self.up_knn = UpsampleBlock()
        self.up = torch.nn.ModuleList()
        for layer_n in range(num_enc_layers):
            self.lin.append(nn.Linear(in_dims[layer_n], out_dim))
            self.up.append(
                GridSampleConv(
                    in_fdim=out_dim,
                    out_fdim=out_dim,
                    subsampling_dist=-1,
                    kernel_radius=layer_kernel_radii[layer_n],
                    preactivate=True,
                    kernel_debug_viz=kernel_debug_viz,
                    layernorm=False,
                    grid_sampling_method=grid_sampling_method,
                    # sampling_algo=sampling_algo,
                )
            )

        self.mlp = DecreasingMLP(4, out_dim, output_size, final_layer_norm=False)

    def forward(self, points, feats, masks, smpl_ids=None):
        _, query_features = self.up_knn(
            points[-2], points[-1], feats[-1], q_mask=masks[-2], t_mask=masks[-1]
        )
        pe, fe, _ = self.up[-1](
            src_points=points[-1],
            query_points=points[-2],
            src_features=feats[-1],
            query_features=query_features,
            src_mask=masks[-1],
            query_mask=masks[-1],
        )
        # print(points[-1].shape)
        # print(pe.shape)
        fe = fe + feats[-2]  # skip connection
        for layer_n in range(self.num_enc_layers - 2, 0, -1):
            _, query_features = self.up_knn(
                points[layer_n],
                pe,
                fe,
                q_mask=masks[layer_n],
                t_mask=masks[layer_n + 1],
            )
            pe, fe, _ = self.up[layer_n](
                src_points=pe,
                query_points=points[layer_n],
                src_features=fe,
                query_features=query_features,
                src_mask=masks[layer_n + 1],
                query_mask=masks[layer_n],
            )
            if self.enable_dropout:
                fe = self.dropout(fe)
            # print(fe.shape, feats[layer_n].shape)
            fe = fe + self.lin[layer_n](feats[layer_n])  # skip connection

        _, query_features = self.up_knn(points[0], pe, fe, q_mask=None, t_mask=masks[0])
        pe, fe, _ = self.up[0](
            src_points=pe,
            query_points=points[0],
            src_features=fe,
            query_features=query_features,
            src_mask=masks[0],
            query_mask=None,
        )
        if self.enable_dropout:
            fe = self.dropout(fe)
        # skip connection
        fe = fe + self.lin[0](feats[0])

        if self.pos_enc:
            fe = torch.cat((fe, points), dim=-1)

        # MLP
        leaf_offs = self.mlp(fe)

        return leaf_offs


class OffsetModel(nn.Module):
    def __init__(self, cfg):

        super().__init__()

        # Read config
        self.cfg = cfg
        self.enable_leaves = self.cfg["enable_leaves"]
        self.enable_plants = self.cfg["enable_plants"]
        self.min_kernel_r = self.cfg["min_kernel_r"]
        self.max_kernel_r = self.cfg["max_kernel_r"]
        self.num_enc_layers = self.cfg["num_enc_layers"]
        self.n_downsamplings = int(torch.tensor(self.cfg["enc_layer_downsample"]).sum())
        self.enc_layer_downsample = self.cfg["enc_layer_downsample"]

        self.max_feat_size = self.cfg["max_feat_size"]
        # input size is color (3) + poses (3) if enabled
        self.input_size = 6 if self.cfg["use_poses_as_feats"] else 3
        self.output_size = 3
        self.conv_r2s_gain = self.cfg["conv_r2s_gain"]
        self.min_feat_size = self.cfg["min_feat_size"]

        self.pos_enc = self.cfg["pos_enc"]
        self.decoder_type = self.cfg["decoder_type"]

        self.enable_dropout = self.cfg["enable_dropout"]
        self.grid_sampling_method = self.cfg["grid_sampling_method"]
        self.sampling_algo = "hashing"

        if "layer_norm" not in cfg.keys():
            self.layer_norm = False
        else:
            self.layer_norm = self.cfg["layer_norm"]

        self.define_model()

    def define_model(self):
        # Compute dependent params
        kernel_radius_range = self.max_kernel_r - self.min_kernel_r
        layer_radius_step = kernel_radius_range / (self.n_downsamplings)
        kernel_feat_range = self.max_feat_size - self.min_feat_size
        layer_feat_step = int(kernel_feat_range / (self.n_downsamplings))

        # Encoder
        self.down = torch.nn.ModuleList()
        kernel_r = self.min_kernel_r
        in_fdim = self.min_feat_size
        out_fdim = in_fdim
        layer_in_fdims = []
        layer_out_fdims = []
        layer_subsampling_dists = []
        layer_kernel_radii = []
        for layer_n in range(self.num_enc_layers):
            # set in feature dim to previous output dim
            in_fdim = out_fdim
            downsample = self.enc_layer_downsample[layer_n]
            if downsample:
                # do downsampling, therefore increase feat size and increase downsampling radius
                out_fdim = in_fdim + layer_feat_step
                # kernel_r += layer_radius_step
            else:
                # no downsampling
                out_fdim = in_fdim
            layer_in_fdims.append(in_fdim)
            layer_out_fdims.append(out_fdim)
            layer_subsampling_dists.append(kernel_r / self.conv_r2s_gain)
            layer_kernel_radii.append(kernel_r)
            self.down.append(
                GridSampleConv(
                    in_fdim=self.input_size if layer_n == 0 else in_fdim,
                    out_fdim=out_fdim,
                    subsampling_dist=kernel_r / self.conv_r2s_gain
                    if (downsample or self.enc_layer_downsample[layer_n - 1])
                    else -1,
                    kernel_radius=kernel_r,
                    preactivate=(layer_n > 0),
                    kernel_debug_viz=False,
                    layernorm=self.layer_norm,
                    grid_sampling_method=self.grid_sampling_method,
                    # sampling_algo=self.sampling_algo,
                )
            )
            # print(in_fdim, out_fdim, kernel_r)
            if downsample:
                # do downsampling, therefore increase feat size and increase downsampling radius
                kernel_r += layer_radius_step

        self.dropout = nn.Dropout(p=0.2)

        # Leaf decoder
        if self.enable_leaves:
            self.leaf_decoder = Decoder(
                self.num_enc_layers,
                self.output_size,
                [layer.in_fdim for layer in self.down],
                self.down[-1].out_fdim,
                layer_kernel_radii,
                self.grid_sampling_method,
                pos_enc=self.pos_enc,
                kernel_debug_viz=False,
                sampling_algo=self.sampling_algo,
            )

        # Plant decoder
        if self.enable_plants:
            self.plant_decoder = Decoder(
                self.num_enc_layers,
                self.cfg["plant_offset_size"],
                [layer.in_fdim for layer in self.down],
                self.down[-1].out_fdim,
                layer_kernel_radii,
                self.grid_sampling_method,
                pos_enc=self.pos_enc,
                kernel_debug_viz=False,
                sampling_algo=self.sampling_algo,
            )

    def forward(self, points, features, input_mask=None):
        # Encoder
        p1, f1, m1 = self.down[0](points, features, src_mask=input_mask)
        p = []
        f = []
        m = []
        s = []
        p.append(p1)
        f.append(f1)
        m.append(m1)
        # s.append(s1)
        
        # print(points.shape)
        # print(p1.shape)

        for layer_n in range(1, self.num_enc_layers):
            p_new, f_new, m_new = self.down[layer_n](
                p[layer_n - 1], f[layer_n - 1], src_mask=m[layer_n - 1]
            )
            if self.enable_dropout:
                f_new = self.dropout(f_new)
            p.append(p_new)
            f.append(f_new)
            m.append(m_new)
            # s.append(s_new)
            # print(p_new.shape)
        # print("Encoder finished")
        # Leaf decoder
        if self.enable_leaves:
            leaf_offs = self.leaf_decoder.forward(
                [
                    points,
                ]
                + p,
                [
                    features,
                ]
                + f,
                [
                    input_mask,
                ]
                + m,
                smpl_ids=s,
            )
        else:
            leaf_offs = None

        # Plant decoder
        if self.enable_plants:
            plant_offs = self.plant_decoder.forward(
                [
                    points,
                ]
                + p,
                [
                    features,
                ]
                + f,
                [
                    input_mask,
                ]
                + m,
                smpl_ids=s,
            )
        else:
            plant_offs = None

        return leaf_offs, plant_offs
