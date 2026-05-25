from utils2 import *
from propagation_ASM import *
from torch import Tensor
from complexCNN import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


def haar_dwt2d(image: Tensor):
    """
    (B, C, H, W)
    LL, LH, HL, HH, each (B, C, H/2, W/2)
    """
    b, c, h, w = image.shape
    ll_filter = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
    lh_filter = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2.0
    hl_filter = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2.0
    hh_filter = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
    filters = torch.stack([ll_filter, lh_filter, hl_filter, hh_filter]).unsqueeze(1)
    filters = filters.to(dtype=image.dtype, device=image.device)
    filters = filters.repeat(c, 1, 1, 1)

    out = F.conv2d(image, filters, stride=2, groups=c)
    out = out.view(b, c, 4, out.shape[2], out.shape[3])
    LL = out[:, :, 0, :, :]
    LH = out[:, :, 1, :, :]
    HL = out[:, :, 2, :, :]
    HH = out[:, :, 3, :, :]
    return LL, LH, HL, HH


def haar_idwt2d(LL: Tensor, LH: Tensor, HL: Tensor, HH: Tensor):
    """
    LL, LH, HL, HH, each (B, C, H/2, W/2)
    reconstructed image (B, C, H, W)
    """
    b, c, h_half, w_half = LL.shape
    x = torch.stack([LL, LH, HL, HH], dim=2)
    x = x.view(b, c * 4, h_half, w_half)

    ll_filter = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
    lh_filter = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2.0
    hl_filter = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2.0
    hh_filter = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
    filters = torch.stack([ll_filter, lh_filter, hl_filter, hh_filter]).unsqueeze(1)
    filters = filters.to(dtype=LL.dtype, device=LL.device)
    filters = filters.repeat(c, 1, 1, 1)
    reconstructed_image = F.conv_transpose2d(x, filters, stride=2, groups=c)
    return reconstructed_image


def complex_haar_dwt2d(image: Tensor):

    if image.is_complex():
        real_part = image.real
        imag_part = image.imag

        LL_r, LH_r, HL_r, HH_r = haar_dwt2d(real_part)
        LL_i, LH_i, HL_i, HH_i = haar_dwt2d(imag_part)

        LL = torch.complex(LL_r, LL_i)
        LH = torch.complex(LH_r, LH_i)
        HL = torch.complex(HL_r, HL_i)
        HH = torch.complex(HH_r, HH_i)
        return LL, LH, HL, HH
    else:
        return haar_dwt2d(image)


def complex_haar_idwt2d(LL: Tensor, LH: Tensor, HL: Tensor, HH: Tensor):

    if LL.is_complex():
        real_reconstructed = haar_idwt2d(LL.real, LH.real, HL.real, HH.real)
        imag_reconstructed = haar_idwt2d(LL.imag, LH.imag, HL.imag, HH.imag)
        return torch.complex(real_reconstructed, imag_reconstructed)
    else:
        return haar_idwt2d(LL, LH, HL, HH)


def pad_shape_to_multiple(h: int, w: int, multiple: int = 64):
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    return new_h, new_w


def pad_to_multiple(x: torch.Tensor, multiple: int = 64, mode: str = "replicate"):
    
    if x.dim() != 4:
        raise ValueError(f"pad_to_multiple expects a BCHW tensor, got shape {tuple(x.shape)}")

    _, _, h, w = x.shape
    new_h, new_w = pad_shape_to_multiple(h, w, multiple=multiple)
    pad_h = new_h - h
    pad_w = new_w - w

    if pad_h == 0 and pad_w == 0:
        return x, h, w

    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, h, w



class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, ch), num_channels=ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class FastPhaseStudent(nn.Module):
    """
    轻量高频相位 student 网络
    """

    def __init__(self, in_ch=3, out_ch=3, base_ch=32, num_res_blocks=4, depth_embed_dim=0):
        super().__init__()
        self.depth_embed_dim = depth_embed_dim

        self.stem = ConvGNAct(in_ch, base_ch)
        self.res_blocks = nn.ModuleList([ResidualBlock(base_ch) for _ in range(num_res_blocks)])
        self.refine = nn.Sequential(
            ConvGNAct(base_ch, base_ch),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1, bias=True),
        )

        if depth_embed_dim > 0:
            self.film_layers = nn.ModuleList([
                nn.Linear(depth_embed_dim, base_ch * 2) 
                for _ in range(num_res_blocks)
            ])

    def forward(self, x: Tensor, depth_emb: Tensor = None):
        """
        x:         [B, C, H, W]  coarse high-frequency input
        depth_emb: [B, depth_embed_dim]
        """
        feat = self.stem(x)

        for i, block in enumerate(self.res_blocks):
            feat = block(feat)
  
            if depth_emb is not None and self.depth_embed_dim > 0:
                film = self.film_layers[i](depth_emb)
                scale, shift = film.chunk(2, dim=-1)
                scale = scale[:, :, None, None]
                shift = shift[:, :, None, None]
                feat = feat * (1.0 + scale) + shift

        return self.refine(feat)


class FDGNet(nn.Module):
    def __init__(
        self,
        cldm,                           # teacher latent diffusion model (SD V2.1 + ControlNet)
        size,
        feature_size=8e-6,
        distance_range=0.03,
        img_distance=0.2,
        layers_num=30,
        wavelengths=None, 
        depth_embed_dim=128,
        context_dim=1024,
        student_base_ch=32,
        student_res_blocks=4,
        pad_multiple=64,
        use_teacher_for_distill=True,
        mode='train',
    ):
        super().__init__()

        self.cldm = cldm
        self.use_teacher_for_distill = use_teacher_for_distill
        self.freeze_teacher = True
        self.mode = mode
        self.pad_multiple = pad_multiple
        self.context_dim = context_dim

        self.wavelengths = wavelengths if wavelengths is not None else [632e-9, 520e-9, 450e-9]
        self.feature_size = [feature_size, feature_size]
        self.distance_range = distance_range
        self.img_distance = img_distance
        self.layers_num = layers_num

        if self.freeze_teacher and hasattr(self.cldm, 'eval'):
            self.cldm.eval()
            for p in self.cldm.parameters():
                p.requires_grad_(False)

        self.num_depths = layers_num
        self.depth_embed_dim = depth_embed_dim
        self.depth_embedding = nn.Embedding(layers_num, depth_embed_dim)
        self.depth_context_proj = nn.Linear(depth_embed_dim, context_dim)


        self.size = list(self._pad_shape(size[0], size[1]))

        if self.mode == 'train':
            self.pre_kernel = []
            self.pre_kernel_inv = []
            for i in track(range(layers_num)):
                distance = (0 - self.distance_range) / self.layers_num * i
                dis = distance - self.img_distance
                if isinstance(dis, torch.Tensor):
                    dis = dis.item()
                dis = round(dis, 6)

                kernel = propagation_ASM(
                    torch.empty(1, 3, self.size[0], self.size[1]),
                    feature_size=self.feature_size,
                    wavelength=self.wavelengths,
                    z=dis,
                    return_H=True
                ).to('cuda').detach()
                kernel.requires_grad = False
                self.pre_kernel.append(kernel)

                kernel_inv = propagation_ASM(
                    torch.empty(1, 3, self.size[0], self.size[1]),
                    feature_size=self.feature_size,
                    wavelength=self.wavelengths,
                    z=-dis,
                    return_H=True
                ).to('cuda').detach()
                kernel_inv.requires_grad = False
                self.pre_kernel_inv.append(kernel_inv)
        else:
            self.H_fwd = propagation_ASM(
                torch.empty(1, 3, self.size[0], self.size[1]),
                feature_size=self.feature_size,
                wavelength=self.wavelengths,
                z=self.img_distance,
                return_H=True
            ).to('cuda').detach()
            self.H_fwd.requires_grad = False

            self.H_bwd = propagation_ASM(
                torch.empty(1, 3, self.size[0], self.size[1]),
                feature_size=self.feature_size,
                wavelength=self.wavelengths,
                z=-self.img_distance,
                return_H=True
            ).to('cuda').detach()
            self.H_bwd.requires_grad = False

        self.network1 = ComplexCNN1()

        self.student_hf_net = FastPhaseStudent(
            in_ch=3,
            out_ch=3,
            base_ch=student_base_ch,
            num_res_blocks=student_res_blocks,
            depth_embed_dim=depth_embed_dim,
        )
        self.aa = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def _pad_shape(self, h, w):
        new_h = (h + self.pad_multiple - 1) // self.pad_multiple * self.pad_multiple
        new_w = (w + self.pad_multiple - 1) // self.pad_multiple * self.pad_multiple
        return new_h, new_w

    def _pad_input(self, x: Tensor):
        B, C, H, W = x.shape
        new_H, new_W = self._pad_shape(H, W)
        x = F.pad(x, (0, new_W - W, 0, new_H - H), mode='replicate')
        return x, H, W

    def _amp_context(self):
        if torch.cuda.is_available():
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()

    def _build_hf_coarse(self, U_proj: Tensor) -> Tensor:
        """
        两级 Haar DWT，提取高频残差
        """
        phase_proj = torch.angle(U_proj)

        LL1, LH1, HL1, HH1 = haar_dwt2d(phase_proj)
        LL2, LH2, HL2, HH2 = haar_dwt2d(LL1)

        zeros_LL2 = torch.zeros_like(LL2)
        hf2_at_level1 = haar_idwt2d(zeros_LL2, LH2, HL2, HH2)

        attenuated_HH1 = HH1 * 0.05
        hf_coarse = haar_idwt2d(hf2_at_level1, LH1, HL1, attenuated_HH1)
        return hf_coarse

    def _physics_projection(self, source: Tensor, ikk: int) -> Tensor:
        """
        物理反向传播 + 近端投影，得到 U_proj
        """
        H_fwd = self.pre_kernel[ikk]     if self.mode == 'train' else self.H_fwd
        H_bwd = self.pre_kernel_inv[ikk] if self.mode == 'train' else self.H_bwd

        target_field = torch.complex(source, torch.zeros_like(source))
        slm_init = propagation_ASM(
            target_field, self.feature_size, self.wavelengths, z=None, precomped_H=H_bwd
        )
        Ax = propagation_ASM(
            slm_init, self.feature_size, self.wavelengths, z=None, precomped_H=H_fwd
        )
        u = (torch.abs(Ax) - source) * torch.exp(1j * torch.angle(Ax))
        grad = propagation_ASM(
            u, self.feature_size, self.wavelengths, z=None, precomped_H=H_bwd
        )
        return slm_init - grad


    @torch.no_grad()
    def forward_teacher(self, hf_coarse: Tensor, z_m: Tensor) -> Tensor:
        """
        离线生成 teacher 高频相位 target，供蒸馏训练使用
        """
        device = hf_coarse.device
        B = hf_coarse.shape[0]

        depth_emb = self.depth_embedding(z_m)                # [B, depth_embed_dim]
        depth_token = self.depth_context_proj(depth_emb)     # [B, context_dim]
        depth_token = depth_token[:, None, :]                # [B, 1, context_dim]


        cond = self.cldm.prepare_condition(hf_coarse)
        x_start = self.cldm.vae_encode(hf_coarse)

        T = 50
        t = torch.randint(0, T, (B,), device=device).long()
        betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, T,
                               dtype=torch.float32, device=device) ** 2
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        sqrt_alpha     = torch.sqrt(alphas_cumprod[t]).view(B, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod[t]).view(B, 1, 1, 1)
        noise   = torch.randn_like(x_start)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus * noise

        context_tensor = None
        hint_tensor    = None
        if isinstance(cond, dict):
            c_txt = cond.get('c_crossattn', cond.get('context', None))
            if c_txt is not None:
                context_tensor = torch.cat(c_txt, 1) if isinstance(c_txt, list) else c_txt
            c_img = cond.get('c_concat', None)
            if c_img is not None:
                hint_tensor = torch.cat(c_img, 1) if isinstance(c_img, list) else c_img
        else:
            context_tensor = cond

        if context_tensor is None:
            # 无文本条件：全零占位（77 tokens，SD V2.1 默认序列长度）
            context_tensor = torch.zeros(
                (B, 77, self.context_dim), dtype=x_noisy.dtype, device=device
            )

        depth_token    = depth_token.to(dtype=context_tensor.dtype)
        context_tensor = torch.cat([context_tensor, depth_token], dim=1)  # [B, 78, 1024]

        control_res  = None
        control_net  = getattr(self.cldm, 'control_model',
                               getattr(self.cldm, 'controlnet', None))
        if control_net is not None and hint_tensor is not None:
            control_res = control_net(
                x=x_noisy, hint=hint_tensor, timesteps=t, context=context_tensor
            )
            if hasattr(self.cldm, 'control_scales'):
                control_res = [c * s for c, s in zip(control_res, self.cldm.control_scales)]


        pred_noise    = self.cldm.unet(
            x=x_noisy, timesteps=t, context=context_tensor, control=control_res
        )
        pred_x0       = (x_noisy - sqrt_one_minus * pred_noise) / sqrt_alpha
        pred_hf_phase = self.cldm.vae_decode(pred_x0)
        return pred_hf_phase


    def forward(self, source: Tensor, ikk, return_aux=False):

        if isinstance(ikk, torch.Tensor):
            ikk = ikk.item()
        ikk = int(ikk)

        source, orig_H, orig_W = self._pad_input(source)

        with torch.no_grad():
            U_proj    = self._physics_projection(source, ikk)
            hf_coarse = self._build_hf_coarse(U_proj)

        device = source.device
        B      = source.shape[0]
        z_m    = torch.full((B,), ikk, dtype=torch.long, device=device)
        depth_emb = self.depth_embedding(z_m)   # [B, depth_embed_dim]，有梯度

        with self._amp_context():
            pred_hf_phase = self.student_hf_net(hf_coarse, depth_emb=depth_emb)

        pred_lf_phase = self.network1(U_proj)

        full_phase = pred_hf_phase + self.aa * pred_lf_phase

        H_fwd = self.pre_kernel[ikk] if self.mode == 'train' else self.H_fwd
        slm_r, slm_i = polar_to_rect(torch.ones_like(full_phase), full_phase)
        slm_field    = torch.complex(slm_r, slm_i)
        recon_field  = propagation_ASM(
            slm_field, self.feature_size, self.wavelengths, z=None, precomped_H=H_fwd
        )
        recon_amp = torch.abs(recon_field)

        recon_amp = recon_amp[:, :, :orig_H, :orig_W]

        if return_aux:
            aux = {
                'hf_coarse':     hf_coarse[:, :, :orig_H, :orig_W],
                'pred_hf_phase': pred_hf_phase[:, :, :orig_H, :orig_W],
                'pred_lf_phase': pred_lf_phase[:, :, :orig_H, :orig_W],
                'full_phase':    full_phase[:, :, :orig_H, :orig_W],
                'depth_emb':     depth_emb,
                'U_proj':        U_proj,
            }
            return recon_amp, aux

        return recon_amp

    def distill_loss(self, student_pred: Tensor, teacher_pred: Tensor, hf_coarse: Tensor = None):
        loss_l1 = F.l1_loss(student_pred, teacher_pred)
        loss_l2 = F.mse_loss(student_pred, teacher_pred)
        loss = loss_l1 + 0.5 * loss_l2

        if hf_coarse is not None:
            # 保守的高频一致性约束：让 student 输出的高频统计接近 teacher
            s_LL, s_LH, s_HL, s_HH = haar_dwt2d(student_pred)
            t_LL, t_LH, t_HL, t_HH = haar_dwt2d(teacher_pred)
            loss_freq = (
                F.l1_loss(s_LH, t_LH)
                + F.l1_loss(s_HL, t_HL)
                + F.l1_loss(s_HH, t_HH)
            )
            loss = loss + 0.2 * loss_freq

        return loss