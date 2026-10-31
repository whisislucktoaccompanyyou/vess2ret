"""Modern TensorFlow/Keras 3 model definitions for the vess2ret project.

The original repository used Python 2 and an old Keras/Theano channels-first API.
This version keeps the public functions compatible while using TensorFlow's
channels-last convention and current Keras APIs.
"""

import keras
from keras import ops
from keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    LeakyReLU,
)
from keras.models import Model
from keras.optimizers import Adam


def concatenate_layers(inputs, concat_axis=-1, mode="concat"):
    """Concatenate tensors along the channels-last channel axis."""
    if mode != "concat":
        raise ValueError("Only concatenation is supported")
    return Concatenate(axis=concat_axis)(inputs)


def Convolution(f, k=3, s=2, border_mode="same", **kwargs):
    """Compatibility wrapper around :class:`keras.layers.Conv2D`."""
    return Conv2D(
        filters=f,
        kernel_size=(k, k),
        padding=border_mode,
        strides=(s, s),
        data_format="channels_last",
        **kwargs,
    )


def Deconvolution(f, output_shape=None, k=2, s=2, **kwargs):
    """Compatibility wrapper around :class:`Conv2DTranspose`.

    ``output_shape`` was required by the old Theano backend. Keras 3 infers
    the spatial output shape from the input tensor, so it is intentionally
    ignored.
    """
    del output_shape
    return Conv2DTranspose(
        filters=f,
        kernel_size=(k, k),
        strides=(s, s),
        padding=kwargs.pop("padding", "same"),
        data_format="channels_last",
        **kwargs,
    )


def BatchNorm(mode=2, axis=-1, **kwargs):
    """Compatibility wrapper around :class:`BatchNormalization`."""
    del mode
    return BatchNormalization(axis=axis, **kwargs)


def g_unet(in_ch, out_ch, nf, batch_size=1, is_binary=False, name="unet"):
    """Create the 512x512 U-Net generator.

    Images use the TensorFlow convention ``(batch, height, width, channels)``.
    ``batch_size`` is retained for compatibility and is not fixed in the input
    shape, which allows Keras to train with different batch sizes.
    """
    del batch_size
    inputs = Input(shape=(512, 512, in_ch), name="input")

    conv1 = LeakyReLU(0.2)(BatchNorm()(Convolution(nf)(inputs)))
    conv2 = LeakyReLU(0.2)(BatchNorm()(Convolution(nf * 2)(conv1)))
    conv3 = LeakyReLU(0.2)(BatchNorm()(Convolution(nf * 4)(conv2)))
    conv4 = LeakyReLU(0.2)(BatchNorm()(Convolution(nf * 8)(conv3)))
    conv5 = LeakyReLU(0.2)(BatchNorm()(Convolution(nf * 8)(conv4)))
    conv6 = LeakyReLU(0.2)(BatchNorm()(Convolution(nf * 8)(conv5)))
    conv7 = LeakyReLU(0.2)(BatchNorm()(Convolution(nf * 8)(conv6)))
    conv8 = LeakyReLU(0.2)(BatchNorm()(Convolution(nf * 8)(conv7)))
    bottleneck = LeakyReLU(0.2)(
        BatchNorm()(Convolution(nf * 8, k=2, s=1, border_mode="valid")(conv8))
    )

    # The old implementation supplied an explicit Theano output_shape for the
    # first transpose convolution. In Keras 3 the equivalent operation must
    # use stride 2 to grow 1x1 back to 2x2.
    x = Dropout(0.5)(BatchNorm()(Deconvolution(nf * 8, k=2, s=2)(bottleneck)))
    x = LeakyReLU(0.2)(Concatenate(axis=-1)([x, conv8]))

    x = Dropout(0.5)(BatchNorm()(Deconvolution(nf * 8)(x)))
    x = LeakyReLU(0.2)(Concatenate(axis=-1)([x, conv7]))

    x = Dropout(0.5)(BatchNorm()(Deconvolution(nf * 8)(x)))
    x = LeakyReLU(0.2)(Concatenate(axis=-1)([x, conv6]))

    x = LeakyReLU(0.2)(Concatenate(axis=-1,)(
        [BatchNorm()(Deconvolution(nf * 8)(x)), conv5]
    ))
    x = LeakyReLU(0.2)(Concatenate(axis=-1)(
        [BatchNorm()(Deconvolution(nf * 8)(x)), conv4]
    ))
    x = LeakyReLU(0.2)(Concatenate(axis=-1)(
        [BatchNorm()(Deconvolution(nf * 4)(x)), conv3]
    ))
    x = LeakyReLU(0.2)(Concatenate(axis=-1)(
        [BatchNorm()(Deconvolution(nf * 2)(x)), conv2]
    ))
    x = LeakyReLU(0.2)(Concatenate(axis=-1)(
        [BatchNorm()(Deconvolution(nf)(x)), conv1]
    ))

    output = Activation("sigmoid" if is_binary else "tanh")(
        Deconvolution(out_ch)(x)
    )
    return Model(inputs, output, name=name)


def discriminator(
    a_ch,
    b_ch,
    nf,
    opt=None,
    name="d",
):
    """Create and compile the PatchGAN discriminator."""
    if opt is None:
        opt = Adam(learning_rate=2e-4, beta_1=0.5)

    inputs = Input(shape=(512, 512, a_ch + b_ch), name="input")
    x = LeakyReLU(0.2)(Convolution(nf)(inputs))
    x = LeakyReLU(0.2)(Convolution(nf * 2)(x))
    x = LeakyReLU(0.2)(Convolution(nf * 4)(x))
    x = LeakyReLU(0.2)(Convolution(nf * 8)(x))
    output = Activation("sigmoid")(Convolution(1)(x))

    model = Model(inputs, output, name=name)
    model.compile(optimizer=opt, loss="binary_crossentropy")
    return model


def pix2pix(
    atob,
    d,
    a_ch,
    b_ch,
    alpha=100,
    is_a_binary=False,
    is_b_binary=False,
    opt=None,
    name="pix2pix",
):
    """Create the generator-training model.

    Keras 3 does not support the old closure-based loss that referenced
    symbolic tensors from outside the loss function. The model therefore has
    two outputs: the adversarial prediction and the generated image. Their
    losses are combined with ``alpha`` as the reconstruction-loss weight.
    """
    del is_a_binary
    if opt is None:
        opt = Adam(learning_rate=2e-4, beta_1=0.5)

    a = Input(shape=(512, 512, a_ch), name="a")
    b = Input(shape=(512, 512, b_ch), name="b")
    generated = atob(a)
    discriminator_input = Concatenate(axis=-1)([a, generated])
    adversarial = d(discriminator_input)

    model = Model([a, b], [adversarial, generated], name=name)
    d.trainable = False
    reconstruction_loss = "binary_crossentropy" if is_b_binary else "mae"
    model.compile(
        optimizer=opt,
        loss=["binary_crossentropy", reconstruction_loss],
        loss_weights=[1.0, alpha],
    )
    return model


if __name__ == "__main__":
    generator = g_unet(1, 3, 1)
    disc = discriminator(1, 3, 1)
    combined = pix2pix(generator, disc, 1, 3)
    generator.summary()
    disc.summary()
    combined.summary()
