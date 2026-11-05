# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets"
#

"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import dnnlib
import legacy

from pg_modules.blocks import Interpolate
import timm
from pg_modules.projector import get_backbone_normstats
from training.sd_loss import SDLoss

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0,
                 train_head_only=False, style_mixing_prob=0.0, pl_weight=0.0,
                 cls_model='efficientnet_b1', cls_weight=0.0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.train_head_only = train_head_only

        # SG2 techniques
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = 2
        self.pl_decay = 0.01
        self.pl_no_weight_grad = True
        self.pl_mean = torch.zeros([], device=device)

        # classifier guidance
        cls = timm.create_model(cls_model, pretrained=True).eval()
        self.classifier = nn.Sequential(Interpolate(224), cls).to(device)
        normstats = get_backbone_normstats(cls_model)
        self.norm = Normalize(normstats['mean'], normstats['std'])
        self.cls_weight = cls_weight
        self.cls_guidance_loss = torch.nn.CrossEntropyLoss()

        #self.sd_loss_weight = 200.0 #kwargs.get('sd_loss', 0.0) # kwargs.get('sd_aug', False)
        #self.sd_ramp_kimg = 300 # kwargs.get('sd_ramp_kimg', self.blur_fade_kimg)
        #loss_type = 'lpips' # 'lpips' # kwargs.get('sd_loss_type', 'lpips')
        #self.sd_loss_module = SDLoss(device, use_aug=True, loss_type=loss_type)
        
        self.sd_loss_weight = kwargs.get('sd_loss', 0.0)
        self.sd_ramp_kimg = self.blur_fade_kimg if kwargs.get('sd_ramp', True) else 0
        loss_type = kwargs.get('sd_loss_type', 'lpips')
        sd_aug = kwargs.get('sd_aug', True)
        self.sd_loss_module = SDLoss(device, use_aug=sd_aug, loss_type=loss_type)
        
        print("sd_loss_weight", self.sd_loss_weight, "sd_ramp_kimg", self.sd_ramp_kimg, loss_type)

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=False)  # enabling emas leads to collapse with PG
        return img, ws

    def run_G_ema(self, z, c, update_emas=False):
        ws = self.G_ema.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G_ema.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G_ema.synthesis(ws, update_emas=False)  # enabling emas leads to collapse with PG
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        return self.D(img, c)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg']: return  # no regularization

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        # Gmain: Maximize logits for generated images.
        if do_Gmain:

            # disable gradients for superres
            if self.train_head_only:
                self.G.mapping.requires_grad_(False)
                for name in self.G.synthesis.layer_names:
                    getattr(self.G.synthesis, name).requires_grad_(name in self.G.head_layer_names)

            with torch.autograd.profiler.record_function('Gmain_forward'):
                # Save RNG state so run_G_ema sees identical random noise as run_G
                pre_g_rng_cpu = torch.random.get_rng_state()
                pre_g_rng_cuda = torch.cuda.get_rng_state(gen_z.device)

                gen_img, _gen_ws = self.run_G(gen_z, gen_c)

                # Save RNG state after run_G so downstream operations (like run_D) are not corrupted
                post_g_rng_cpu = torch.random.get_rng_state()
                post_g_rng_cuda = torch.cuda.get_rng_state(gen_z.device)

                with torch.no_grad():
                    torch.random.set_rng_state(pre_g_rng_cpu)
                    torch.cuda.set_rng_state(pre_g_rng_cuda, gen_z.device)
                    gen_img_ema, _ = self.run_G_ema(gen_z, gen_c)

                # Restore RNG state to post-G state
                torch.random.set_rng_state(post_g_rng_cpu)
                torch.cuda.set_rng_state(post_g_rng_cuda, gen_z.device)

                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)

                loss_Gmain = sum([(-l).mean() for l in gen_logits])
                gen_logits = torch.cat(gen_logits)

                loss_G_sd = self.sd_loss_module(gen_img, gen_img_ema)

                if self.cls_weight:
                    gen_img = self.norm(gen_img.add(1).div(2)) 
                    guidance_loss = self.cls_guidance_loss(self.classifier(gen_img), gen_c.argmax(1))
                    loss_Gmain += self.cls_weight * guidance_loss
                    training_stats.report('Loss/G/guidance_loss', guidance_loss)

                sd_ramp = max(min((cur_nimg - self.sd_ramp_kimg * 1e3) / (self.sd_ramp_kimg * 1e3), 1.0), 0.0) if self.sd_ramp_kimg > 0 else 1.0
                sd_weight = self.sd_loss_weight * sd_ramp

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)
                training_stats.report('Loss/G/loss_sd', loss_G_sd)
                training_stats.report('Loss/G/sd_weight', sd_weight)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                if sd_weight > 0:
                    (loss_Gmain + loss_G_sd * sd_weight).backward()
                else:
                    loss_Gmain.backward()

        # Gpl: Apply path length regularization.
        start_plreg = (cur_nimg >= 1e6)
        if start_plreg and self.pl_weight and phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().backward()

        # Dmain: Minimize logits for generated images.
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in gen_logits])
                gen_logits = torch.cat(gen_logits)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            name = 'Dreal'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = sum([(F.relu(torch.ones_like(l) - l)).mean() for l in real_logits])
                real_logits = torch.cat(real_logits)

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
            with torch.autograd.profiler.record_function(name + '_backward'):
                loss_Dreal.backward()
