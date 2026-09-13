from tinyvlm.train_with_validation import corpus_bleu, split_by_image


def test_image_level_split_has_no_leakage():
    samples = [
        (f"image_{image}.jpg", f"caption {caption}")
        for image in range(10)
        for caption in range(5)
    ]
    train_indices, val_indices, train_images, val_images = split_by_image(
        samples, val_ratio=0.1, seed=7
    )

    assert train_images.isdisjoint(val_images)
    assert len(train_images) == 9
    assert len(val_images) == 1
    assert len(train_indices) == 45
    assert len(val_indices) == 5


def test_corpus_bleu_is_one_for_exact_match():
    scores = corpus_bleu(
        ["a dog runs outside"],
        [["a dog runs outside", "the dog is running"]],
    )
    assert scores["bleu1"] == 1.0
    assert scores["bleu4"] == 1.0
