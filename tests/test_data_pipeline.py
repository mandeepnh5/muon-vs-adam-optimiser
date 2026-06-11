"""GPU-resident augmentation pipeline: every transform is verified in
isolation on synthetic data (CPU; no download needed), plus determinism,
label pairing, and exact normalization."""

import torch

from muon_bench import CIFAR10_MEAN, CIFAR10_STD, GPUCifar10

N_TRAIN, N_TEST, BS = 64, 32, 16


def synth(seed=0, constant=None):
    """Synthetic CIFAR-shaped data. If `constant`, image i is filled with
    value constant[i] so identity can be recovered from any pixel."""
    gen = torch.Generator().manual_seed(seed)
    if constant is not None:
        train_x = torch.stack([torch.full((3, 32, 32), v, dtype=torch.uint8) for v in constant])
    else:
        train_x = torch.randint(0, 256, (N_TRAIN, 3, 32, 32), generator=gen, dtype=torch.uint8)
    train_y = (torch.arange(N_TRAIN) * 7) % 10
    test_x = torch.randint(0, 256, (N_TEST, 3, 32, 32), generator=gen, dtype=torch.uint8)
    test_y = torch.arange(N_TEST) % 10
    return train_x, train_y, test_x, test_y


def pipe(data, **kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("batch_size", BS)
    kw.setdefault("dtype", torch.float32)
    kw.setdefault("seed", 0)
    return GPUCifar10(*data, **kw)


def normalize_ref(x_uint8):
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1) * 255
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1) * 255
    return (x_uint8.float() - mean) * (1.0 / std)


def test_shapes_dtype_steps():
    p = pipe(synth())
    assert p.steps_per_epoch == N_TRAIN // BS
    batches = list(p.train_batches())
    assert len(batches) == p.steps_per_epoch
    for x, y in batches:
        assert x.shape == (BS, 3, 32, 32) and x.dtype == torch.float32
        assert y.shape == (BS,) and y.dtype == torch.int64
        assert x.is_contiguous(memory_format=torch.channels_last)
        assert torch.isfinite(x).all()


def test_no_augmentation_is_exact_normalization():
    """With pad=0, flip off, cutout off, train batches must be exactly the
    normalized originals -- this pins down the normalization constants."""
    data = synth(seed=1)
    p = pipe(data, pad=0, flip=False, cutout=0)
    expected = normalize_ref(data[0])
    for x, y in p.train_batches():
        # train order is shuffled -> match each served image by content,
        # then check the served label belongs to the matched source image.
        for j in range(x.size(0)):
            match = (expected - x[j]).abs().amax(dim=(1, 2, 3)) < 1e-5
            assert match.any(), "served image is not a normalized original"
            assert y[j] == data[1][match.nonzero()[0, 0]]
    # test batches are unshuffled -> direct comparison.
    out = torch.cat([x for x, _ in p.test_batches(batch_size=10)])
    assert torch.allclose(out, normalize_ref(data[2]), atol=1e-5)


def test_label_pairing_under_augmentation():
    """Constant-valued images: the crop center always lands inside the
    original image, so image identity is recoverable from the center pixel
    and must agree with the served label, even with crop+flip active."""
    values = list(range(N_TRAIN))  # image i filled with value i
    data = synth(constant=values)
    p = pipe(data, pad=4, flip=True, cutout=0)
    mean0, std0 = CIFAR10_MEAN[0] * 255, CIFAR10_STD[0] * 255
    for x, y in p.train_batches():
        v = (x[:, 0, 16, 16] * std0 + mean0).round().long()  # recover image id
        assert torch.equal(y, (v * 7) % 10)


def test_random_crop_stays_in_bounds():
    """Every crop of an all-255 image contains only {0 (zero padding), 255}."""
    data = synth(constant=[255] * N_TRAIN)
    p = pipe(data, pad=4, flip=False, cutout=0)
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1) * 255
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1) * 255
    pad_val = (0 - mean) / std
    img_val = (255 - mean) / std
    for x, _ in p.train_batches():
        for c in range(3):
            xc = x[:, c]
            is_pad = torch.isclose(xc, pad_val[c], atol=1e-4)
            is_img = torch.isclose(xc, img_val[c], atol=1e-4)
            assert (is_pad | is_img).all()
        # the 24x24 center region can never touch padding (offsets <= 8)
        assert torch.isclose(x[:, :, 8:24, 8:24], img_val.expand(BS, 3, 16, 16), atol=1e-4).all()


def test_horizontal_flip():
    """pad=0, cutout=0: each served image equals the normalized original or
    its mirror; with 64 random images both outcomes must occur."""
    data = synth(seed=2)
    p = pipe(data, pad=0, flip=True, cutout=0)
    expected = normalize_ref(data[0])
    flipped, unflipped = 0, 0
    for x, y in p.train_batches():
        for j in range(x.size(0)):
            # recover source index via the label trick is ambiguous (labels
            # repeat), so search by content instead.
            match_plain = (expected - x[j]).abs().amax(dim=(1, 2, 3)) < 1e-4
            match_flip = (expected.flip(-1) - x[j]).abs().amax(dim=(1, 2, 3)) < 1e-4
            assert match_plain.any() or match_flip.any()
            flipped += int(match_flip.any() and not match_plain.any())
            unflipped += int(match_plain.any() and not match_flip.any())
    assert flipped > 0 and unflipped > 0


def test_cutout_erases_one_rectangle():
    """pad=0, flip off: the zeroed region is a single rectangle, identical
    across channels, with between 4x4 (corner-clipped) and 8x8 pixels."""
    data = synth(constant=[200] * N_TRAIN)  # 200 != channel mean -> no fake zeros
    p = pipe(data, pad=0, flip=False, cutout=8)
    for x, _ in p.train_batches():
        for j in range(x.size(0)):
            m = x[j, 0] == 0.0
            assert torch.equal(m, x[j, 1] == 0.0) and torch.equal(m, x[j, 2] == 0.0)
            area = int(m.sum())
            assert 16 <= area <= 64
            rows, cols = m.any(dim=1), m.any(dim=0)
            assert torch.equal(m, rows.unsqueeze(1) & cols.unsqueeze(0))  # solid rectangle


def test_seed_determinism():
    data = synth(seed=3)
    a = [x for x, _ in pipe(data, seed=42).train_batches()]
    b = [x for x, _ in pipe(data, seed=42).train_batches()]
    c = [x for x, _ in pipe(data, seed=43).train_batches()]
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_test_batches_cover_everything_in_order():
    data = synth(seed=4)
    p = pipe(data)
    xs, ys = zip(*p.test_batches(batch_size=10))
    assert torch.allclose(torch.cat(xs), normalize_ref(data[2]), atol=1e-5)
    assert torch.equal(torch.cat(ys), data[3])


def test_vram_accounting():
    p = pipe(synth())
    expected_bytes = N_TRAIN * 3 * 40 * 40 + N_TEST * 3 * 32 * 32 + (N_TRAIN + N_TEST) * 8
    assert abs(p.vram_mb() - expected_bytes / 2**20) < 1e-6
