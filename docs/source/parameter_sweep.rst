Sweeping parameters
========================

You can use the hyperparameter optimization library of your choice in combination with `smack_imc`. Here, we show a simple set up using [wandb](https://wandb.ai/). 

We recommend that parameters are tuned for each image individually, and that maximizing the `score` is used as the tuning objective, instead of `percent_pixels_correct`

Sweep parameters using wandb
**************************************************
.. code-block:: python

    import wandb
    from smack_imc import *

    run = wandb.init(project=<project name>, entity=<user name>)

    args = {
    'IF_image_path': 'example_IF.tiff',
    'IMC_image_path': 'example_IMC.tiff',
    # IF params
    'IF_binarization_threshold': 0.1,
    'IF_gaussian_sigma': 1,
    # IMC params
    'IMC_channel_axis': 2,
    'IMC_arcsinh_normalize': True,
    'IMC_arcsinh_cofactor': 5,
    'IMC_winsorization_lower_limit': None,
    'IMC_winsorization_upper_limit': 0.2 ,
    'IMC_gaussian_sigma': 1,
    'IMC_binarization_threshold': 2,
    # Registration params
    'IF_downscale_axis': 0,
    'registration_max_features': 1000,
    'registration_percentile': 0.2}

    IF_image = imread('IF_image_path')
    IMC_image = imread('IMC_image_path')

    # preprocess IMC image
    IMC_image = preprocess_IMC_nuclear(IMC_image, args.IMC_channel_axis, args.IMC_arcsinh_normalize, args.IMC_arcsinh_cofactor, \
    [args.IMC_winsorization_lower_limit, args.IMC_winsorization_upper_limit],\
    args.IMC_binarization_threshold, args.IMC_gaussian_sigma)

    # preprocess IF image
    IF_image = preprocess_IF_nuclear(IF_image, args.IF_binarization_threshold, args.IF_gaussian_sigma)
    IF_image = approx_scale(IF_image, IMC_image, args.IF_downscale_axis)

    # Get registration matrix
    IF_aligned, M = register(IF_image, IMC_image, max_features=args.registration_max_features, keep_percent=args.registration_percentile)

    plot_registration(IF_aligned, IMC_image)
    plt.show()

    if np.isnan(M).all():
        print('registration failure')
        wandb.log({'score':-np.inf})
        wandb.finish()

    else:
        # Log results
        score, ppc = score_registration(IF_image, IMC_image, M)
        wandb.log({'Score': score, 'Percent pixel corectness': ppc, 'Registration': plot_registration(IF_aligned, IMC_image) })

    wandb.finish()




