"""
Makeup AI Module for AI Beauty Platform

This module provides makeup application and style transfer capabilities
using GAN-based models like BeautyGAN and Makeup Transfer GAN.
"""

from .makeup_transfer import MakeupTransferGAN
from .beauty_gan import BeautyGAN
from .makeup_analyzer import MakeupAnalyzer

__all__ = ['MakeupTransferGAN', 'BeautyGAN', 'MakeupAnalyzer'] 