# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from fairseq2.typing import DataType, Device

from sonar.models.sonar_speech import (
    SonarSpeechEncoderConfig,
    SonarSpeechEncoderFactory,
)
from sonar.models.sonar_text import (
    SonarTextDecoderConfig,
    SonarTextDecoderFactory,
    SonarTextEncoderConfig,
    SonarTextEncoderFactory,
)
from sonar.models.sonar_translation.model import SonarEncoderDecoderModel


def create_sonar_text_encoder_decoder_model(
    encoder_config: SonarTextEncoderConfig,
    decoder_config: SonarTextDecoderConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> SonarEncoderDecoderModel:
    """Create an SonarText EncoderDecoder model.

    :param encoder_config:
        The configuration to use for building SonarTextEncoder.
    :param decoder_config:
        The configuration to use for building SonarTextDecoder.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder = SonarTextEncoderFactory(encoder_config).create_model()
    decoder = SonarTextDecoderFactory(decoder_config).create_model()

    return SonarEncoderDecoderModel(encoder=encoder, decoder=decoder).to(
        device=device, dtype=dtype
    )


def create_sonar_speech_to_text_model(
    encoder_config: SonarSpeechEncoderConfig,
    decoder_config: SonarTextDecoderConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> SonarEncoderDecoderModel:
    """Create an SonarSpeechToText EncoderDecoder model.

    :param encoder_config:
        The configuration to use for building SonarSpeecEncoder.
    :param decoder_config:
        The configuration to use for building SonarTextDecoder.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder = SonarSpeechEncoderFactory(encoder_config).create_model()
    decoder = SonarTextDecoderFactory(decoder_config).create_model()

    return SonarEncoderDecoderModel(encoder=encoder, decoder=decoder).to(
        device=device, dtype=dtype
    )
