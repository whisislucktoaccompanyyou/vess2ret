"""Train the vess2ret pix2pix model with Python 3 and Keras 3."""

import argparse
import math
import os

import numpy as np
from keras.optimizers import Adam
from tqdm import tqdm

import models as m
from util.data import TwoImageIterator
from util.util import (
    MyDict,
    create_expt_dir,
    load_losses,
    load_weights,
    log,
    save_weights,
)


def _loss_value(result):
    if isinstance(result, dict):
        return float(result["loss"])
    if isinstance(result, (list, tuple, np.ndarray)):
        return float(result[0])
    return float(result)


def discriminator_generator(it, atob, dout_size):
    """Yield discriminator inputs and labels indefinitely."""
    while True:
        a, b = next(it)
        fake_b = atob.predict(a, verbose=0)
        fake = np.concatenate((a, fake_b), axis=-1)
        real = np.concatenate((a, b), axis=-1)
        batch_x = np.concatenate((fake, real), axis=0)
        batch_y = np.zeros((batch_x.shape[0],) + tuple(dout_size), dtype=np.float32)
        batch_y[: fake.shape[0]] = 1.0
        yield batch_x, batch_y


def pix2pix_generator(it, dout_size):
    """Yield generator inputs and the two Keras 3 pix2pix targets."""
    while True:
        a, b = next(it)
        adversarial_target = np.zeros((a.shape[0],) + tuple(dout_size), dtype=np.float32)
        yield [a, b], [adversarial_target, b]


def train_discriminator(d, it, samples_per_batch=20):
    """Train the discriminator for the requested number of samples."""
    steps = max(1, math.ceil(samples_per_batch / max(1, getattr(it, "batch_size", 1))))
    values = []
    for _ in range(steps):
        batch_x, batch_y = next(it)
        values.append(_loss_value(d.train_on_batch(batch_x, batch_y, return_dict=True)))
    return float(np.mean(values))


def train_pix2pix(pix2pix, it, samples_per_batch=20):
    """Train the generator for the requested number of samples."""
    steps = max(1, math.ceil(samples_per_batch / max(1, getattr(it, "batch_size", 1))))
    values = []
    for _ in range(steps):
        inputs, targets = next(it)
        values.append(_loss_value(pix2pix.train_on_batch(inputs, targets, return_dict=True)))
    return float(np.mean(values))


def evaluate(models, generators, losses, val_samples=192):
    """Evaluate both models on validation batches."""
    del generators
    d = models.d
    p2p = models.p2p
    it_val = models._validation_iterator
    steps = max(1, math.ceil(val_samples / max(1, it_val.batch_size)))
    d_values = []
    p2p_values = []
    for _ in range(steps):
        a, b = next(it_val)
        fake_b = models.atob.predict(a, verbose=0)
        d_x = np.concatenate((a, fake_b), axis=-1)
        d_y = np.ones((a.shape[0],) + tuple(d.output_shape[1:]), dtype=np.float32)
        d_values.append(_loss_value(d.test_on_batch(d_x, d_y, return_dict=True)))

        p2p_y = np.zeros((a.shape[0],) + tuple(d.output_shape[1:]), dtype=np.float32)
        p2p_values.append(
            _loss_value(p2p.test_on_batch([a, b], [p2p_y, b], return_dict=True))
        )

    d_loss = float(np.mean(d_values))
    p2p_loss = float(np.mean(p2p_values))
    losses["d_val"].append(d_loss)
    losses["p2p_val"].append(p2p_loss)
    print(
        "Train Losses (D={:.5f} / P2P={:.5f}); Validation Losses "
        "(D={:.5f} / P2P={:.5f})".format(
            losses["d"][-1], losses["p2p"][-1], d_loss, p2p_loss
        )
    )
    return d_loss, p2p_loss


def model_creation(d, atob, params):
    """Create the combined generator-training model."""
    opt = Adam(learning_rate=params.lr, beta_1=params.beta_1)
    p2p = m.pix2pix(
        atob,
        d,
        params.a_ch,
        params.b_ch,
        alpha=params.alpha,
        is_a_binary=params.is_a_binary,
        is_b_binary=params.is_b_binary,
        opt=opt,
    )
    # pix2pix() freezes the nested discriminator for generator updates. Re-enable
    # and compile the standalone discriminator for its own update step.
    d.trainable = True
    d.compile(optimizer=Adam(learning_rate=params.lr, beta_1=params.beta_1), loss="binary_crossentropy")
    return MyDict({"atob": atob, "d": d, "p2p": p2p})


def generators_creation(it_train, it_val, models, dout_size):
    """Create compatibility generators used by older callers."""
    generators = MyDict(
        {
            "d_gen": discriminator_generator(it_train, models.atob, dout_size),
            "d_gen_val": discriminator_generator(it_val, models.atob, dout_size),
            "p2p_gen": pix2pix_generator(it_train, dout_size),
            "p2p_gen_val": pix2pix_generator(it_val, dout_size),
        }
    )
    return generators


def train_iteration(models, generators, losses, params):
    """Run one discriminator and one generator update block."""
    d_loss = train_discriminator(models.d, generators["d_gen"], params.samples_per_batch)
    p2p_loss = train_pix2pix(models.p2p, generators["p2p_gen"], params.samples_per_batch)
    losses["d"].append(d_loss)
    losses["p2p"].append(p2p_loss)


def train(models, it_train, it_val, params):
    """Train for the configured number of epochs."""
    create_expt_dir(params)
    dout_size = tuple(models.d.output_shape[1:])
    generators = generators_creation(it_train, it_val, models, dout_size)
    models._validation_iterator = it_val

    train_samples = it_train.N if params.train_samples == -1 else params.train_samples
    batches_per_epoch = max(1, math.ceil(train_samples / params.samples_per_batch))
    val_samples = it_val.N if params.val_samples == -1 else params.val_samples

    losses = {"p2p": [], "d": [], "p2p_val": [], "d_val": []}
    if params.continue_train:
        losses = load_losses(log_dir=params.log_dir, expt_name=params.expt_name)

    for epoch in tqdm(range(params.epochs), desc="训练"):
        for _ in range(batches_per_epoch):
            train_iteration(models, generators, losses, params)
        evaluate(models, generators, losses, val_samples=val_samples)

        if (epoch + 1) % params.save_every == 0:
            save_weights(models, log_dir=params.log_dir, expt_name=params.expt_name)
            log(
                losses,
                models.atob,
                it_val,
                log_dir=params.log_dir,
                expt_name=params.expt_name,
                is_a_binary=params.is_a_binary,
                is_b_binary=params.is_b_binary,
            )


def _bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数只能使用 True 或 False")


def parse_args():
    parser = argparse.ArgumentParser(description="训练 vess2ret 图像到图像生成模型")
    parser.add_argument("--nfd", type=int, default=32)
    parser.add_argument("--nfatob", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--samples_per_batch", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta_1", type=float, default=0.5)
    parser.add_argument("--continue_train", type=_bool, default=False)
    parser.add_argument("--log_dir", default="log")
    parser.add_argument("--expt_name", default=None)
    parser.add_argument("--base_dir", default="data/unet_segmentations_binary")
    parser.add_argument("--train_dir", default="train")
    parser.add_argument("--val_dir", default="val")
    parser.add_argument("--train_samples", type=int, default=-1)
    parser.add_argument("--val_samples", type=int, default=-1)
    parser.add_argument("--load_to_memory", type=_bool, default=True)
    parser.add_argument("--a_ch", type=int, default=1)
    parser.add_argument("--b_ch", type=int, default=3)
    parser.add_argument("--is_a_binary", type=_bool, default=True)
    parser.add_argument("--is_b_binary", type=_bool, default=False)
    parser.add_argument("--is_a_grayscale", type=_bool, default=True)
    parser.add_argument("--is_b_grayscale", type=_bool, default=False)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--rotation_range", type=float, default=0.0)
    parser.add_argument("--height_shift_range", type=float, default=0.0)
    parser.add_argument("--width_shift_range", type=float, default=0.0)
    parser.add_argument("--horizontal_flip", type=_bool, default=False)
    parser.add_argument("--vertical_flip", type=_bool, default=False)
    parser.add_argument("--zoom_range", type=float, default=0.0)
    return MyDict(vars(parser.parse_args()))


def main():
    params = parse_args()
    if params.target_size != 512:
        raise ValueError("当前模型固定输入尺寸为 512；请将 --target_size 设置为 512。")

    dopt = Adam(learning_rate=params.lr, beta_1=params.beta_1)
    unet = m.g_unet(
        params.a_ch,
        params.b_ch,
        params.nfatob,
        batch_size=params.batch_size,
        is_binary=params.is_b_binary,
    )
    d = m.discriminator(params.a_ch, params.b_ch, params.nfd, opt=dopt)
    if params.continue_train:
        load_weights(unet, d, log_dir=params.log_dir, expt_name=params.expt_name)

    target_size = (params.target_size, params.target_size)
    train_dir = os.path.join(params.base_dir, params.train_dir)
    val_dir = os.path.join(params.base_dir, params.val_dir)
    it_train = TwoImageIterator(
        train_dir,
        is_a_binary=params.is_a_binary,
        is_b_binary=params.is_b_binary,
        is_a_grayscale=params.is_a_grayscale,
        is_b_grayscale=params.is_b_grayscale,
        batch_size=params.batch_size,
        load_to_memory=params.load_to_memory,
        rotation_range=params.rotation_range,
        height_shift_range=params.height_shift_range,
        width_shift_range=params.width_shift_range,
        zoom_range=params.zoom_range,
        horizontal_flip=params.horizontal_flip,
        vertical_flip=params.vertical_flip,
        target_size=target_size,
    )
    it_val = TwoImageIterator(
        val_dir,
        is_a_binary=params.is_a_binary,
        is_b_binary=params.is_b_binary,
        is_a_grayscale=params.is_a_grayscale,
        is_b_grayscale=params.is_b_grayscale,
        batch_size=params.batch_size,
        load_to_memory=params.load_to_memory,
        target_size=target_size,
        shuffle=False,
    )
    models = model_creation(d, unet, params)
    train(models, it_train, it_val, params)


if __name__ == "__main__":
    main()
