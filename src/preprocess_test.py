from preprocess import preprocess_one_image

preprocess_one_image(
    "./data/test_tif.tif",
    "./data/preprocess_test_bg20_new",
    shading_sigma=120,
    stripe_sigma=[128, 256],
    stripe_level=7,
    stripe_wavelet='db2',
    bg_radius=20,
    save_intermediates=False,
)
