import blocks
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_points

from .blocks import DecreasingMLP


class ConfidModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Read config
        self.cfg = cfg
        self.min_kernel_r = self.cfg["min_kernel_r"]
        self.max_kernel_r = self.cfg["max_kernel_r"]
        self.num_enc_layers = self.cfg["num_enc_layers"]
        if "enc_layer_downsample" in self.cfg:
            self.n_downsamplings = int(
                torch.tensor(self.cfg["enc_layer_downsample"]).sum()
            )
            self.enc_layer_downsample = self.cfg["enc_layer_downsample"]
        else:
            self.n_downsamplings = self.cfg["n_downsamplings"]
            self.enc_layer_downsample = torch.zeros(self.num_enc_layers)
            self.enc_layer_downsample[: self.n_downsamplings] = 1

        self.max_feat_size = self.cfg["max_feat_size"]
        self.output_size = self.cfg["output_size"]
        self.conv_r2s_gain = self.cfg["conv_r2s_gain"]
        self.min_feat_size = self.cfg["min_feat_size"]

        self.pos_enc = self.cfg["pos_enc"]
        self.final_sigmoid = self.cfg["final_sigmoid"]
        self.mask_injection = self.cfg["mask_injection"]

        if "layer_norm" not in cfg.keys():
            cfg["layer_norm"] = False
        if "grid_sampling_method" in self.cfg.keys():
            self.grid_sampling_method = self.cfg["grid_sampling_method"]
        else:
            self.grid_sampling_method = "mean"
        if "var_pooling" in self.cfg.keys():
            self.var_pooling = self.cfg["var_pooling"]
        else:
            self.var_pooling = False
        if "late_pooling" in self.cfg.keys():
            self.late_pooling = self.cfg["late_pooling"]
        else:
            self.late_pooling = False
        if "enable_dropout" in self.cfg.keys():
            self.enable_dropout = self.cfg["enable_dropout"]
        else:
            self.enable_dropout = False

        if "use_poses_as_feats" in self.cfg.keys():
            self.use_poses_as_feats = self.cfg["use_poses_as_feats"]
        else:
            self.use_poses_as_feats = False

        self.use_emb_feats = self.cfg["use_emb_feats"]

        self.input_size = 3
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
        self.layer_kernel_radii = []
        for layer_n in range(self.num_enc_layers):
            # set in feature dim to previous output dim
            in_fdim = out_fdim
            downsample = self.enc_layer_downsample[layer_n]
            if downsample:
                # do downsampling, therefore increase feat size and increase downsampling radius
                out_fdim = in_fdim + layer_feat_step
                kernel_r += layer_radius_step
            else:
                # no downsampling
                out_fdim = in_fdim
            layer_in_fdims.append(in_fdim)
            layer_out_fdims.append(out_fdim)
            layer_subsampling_dists.append(kernel_r / self.conv_r2s_gain)
            self.layer_kernel_radii.append(kernel_r)
            self.down.append(
                blocks.GridSampleConv(
                    in_fdim=self.input_size if layer_n == 0 else in_fdim,
                    out_fdim=out_fdim,
                    subsampling_dist=kernel_r / self.conv_r2s_gain,
                    # if (downsample or self.enc_layer_downsample[layer_n-1])
                    # else -1,
                    kernel_radius=kernel_r,
                    preactivate=(layer_n > 0),
                    kernel_debug_viz=False,
                    layernorm=self.layer_norm,
                    grid_sampling_method=self.grid_sampling_method,
                )
            )
            print(in_fdim, out_fdim, kernel_r)
            if downsample:
                # do downsampling, therefore increase feat size and increase downsampling radius
                kernel_r += layer_radius_step

        self.dropout = nn.Dropout(p=0.2)
        if self.var_pooling:
            self.embed_mlp = DecreasingMLP(
                4,
                self.down[-1].out_fdim,
                self.down[-1].out_fdim,
                final_layer_norm=False,
            )
            self.final_mlp = DecreasingMLP(2, 1, 1, final_layer_norm=False)
        else:
            self.embed_mlp = DecreasingMLP(
                4, self.down[-1].out_fdim, self.output_size, final_layer_norm=False
            )

    def forward(self, points, features):
        debub_viz = False

        # expand area for confidence estimation around leaf
        confid_expand = 0.1
        if confid_expand > 0:
            filtered_points = points.clone()
            filtered_points[
                ~(features[..., 0].bool())
                .unsqueeze(-1)
                .expand_as(filtered_points)
                .bool()
            ] = 1e6
            dists = knn_points(points, filtered_points, norm=2, K=1).dists[..., 0]
            expanded_mask = features.bool()
            expanded_mask[dists < confid_expand**2] = 1
        else:
            expanded_mask = features.bool()

        # crop points and features to only include points around leaf
        max_samples = expanded_mask[..., 0].sum(dim=-1).max().int()
        cropped_points = torch.zeros(
            (len(points), max_samples, 3), device=points.device
        )
        cropped_feats = torch.zeros(
            (len(points), max_samples, features.shape[-1]), device=points.device
        )
        for batch in range(len(points)):
            curr_mask = expanded_mask[batch, :, 0].bool()
            cropped_points[batch, : curr_mask.sum()] = points[batch, curr_mask]
            cropped_feats[batch, : curr_mask.sum()] = features[batch, curr_mask]

        # Encoder
        p1, f1, m1, in_f1 = self.down[0](
            cropped_points,
            cropped_feats,
            smpl_feats=cropped_feats[..., 0].bool().float().unsqueeze(-1),
        )

        p = []
        f = []
        m = []
        in_f = []
        p.append(p1)
        f.append(f1)
        m.append(m1)
        in_f.append(in_f1)

        if debub_viz:
            full_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(cropped_points[0].detach().cpu())
            )
            colors = cropped_feats[0, :, 0] / cropped_feats[0, :, 0].max()
            colors = torch.stack(
                (torch.zeros_like(colors), colors, torch.zeros_like(colors)), dim=-1
            )
            full_pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu())

        for layer_n in range(1, self.num_enc_layers):
            if self.mask_injection:
                cat_feats = torch.cat((f[layer_n - 1], in_f[layer_n - 1]), dim=-1)
                new_input = self.in_feat_merger[layer_n - 1](cat_feats)
            else:
                new_input = f[layer_n - 1]
            if not (m[layer_n - 1].sum(-2) >= 32).all():
                return None
            assert (m[layer_n - 1].sum(-2) >= 32).all()
            p_new, f_new, m_new, in_f_new = self.down[layer_n](
                src_points=p[layer_n - 1],
                src_features=new_input,
                src_mask=m[layer_n - 1],
                smpl_feats=in_f[layer_n - 1],
            )
            p.append(p_new)
            f.append(f_new)
            m.append(m_new)
            in_f.append(in_f_new.detach())

            if debub_viz:
                viz_src = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(
                        p_new[0][m_new[0].squeeze(-1)].detach().cpu()
                    )
                )
                colors = (
                    in_f_new[0, m_new[0].squeeze(-1), 0]
                    / in_f_new[0, m_new[0].squeeze(-1), 0].max()
                )
                colors = torch.stack(
                    (colors, torch.zeros_like(colors), torch.zeros_like(colors)), dim=-1
                )
                viz_src.colors = o3d.utility.Vector3dVector(colors.detach().cpu())
                o3d.visualization.draw_geometries([full_pcd, viz_src])
                viz_src.paint_uniform_color(np.array((0.0, 0, 1)))

        out_inst = (in_f[-1] > 0).any(dim=-1)
        valid_mask = torch.logical_and(out_inst, m[-1][:, :, 0])
        valid_feats = valid_mask.unsqueeze(-1).expand_as(f[-1]) * f[-1]
        if self.late_pooling:
            valid_feats = self.embed_mlp(valid_feats)

            valid_feat_sum = valid_feats.sum(-2)
            mask_sum = valid_mask.sum(-1)
            mask_sum[mask_sum == 0] = 1
            average_pooled_feat = valid_feat_sum / mask_sum.unsqueeze(-1)
            pred_iou = average_pooled_feat.squeeze()

        elif self.var_pooling:
            valid_feats = self.embed_mlp(valid_feats)

            valid_feat_sum = valid_feats.sum(-2)
            mask_sum = valid_mask.sum(-1)
            mask_sum[mask_sum == 0] = 1
            average_pooled_feat = valid_feat_sum / mask_sum.unsqueeze(-1)
            summed_vars = torch.pow(
                valid_feats - average_pooled_feat.unsqueeze(-2), 2
            ).sum(-2)
            feat_var = summed_vars / mask_sum.unsqueeze(-1)
            pred_iou = self.final_mlp(feat_var.mean(-1).unsqueeze(-1))

        else:
            valid_feat_sum = valid_feats.sum(-2)
            mask_sum = valid_mask.sum(-1)
            mask_sum[mask_sum == 0] = 1
            average_pooled_feat = valid_feat_sum / mask_sum.unsqueeze(-1).expand_as(
                valid_feat_sum
            )

            # MLP
            pred_iou = self.embed_mlp(average_pooled_feat).squeeze()

        if self.final_sigmoid:
            pred_iou = pred_iou.sigmoid()
        else:
            pred_iou = pred_iou.abs()

        return pred_iou
