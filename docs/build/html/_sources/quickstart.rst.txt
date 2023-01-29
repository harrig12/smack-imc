Quickstart
=============

Installation/Usage:
*******************
As the package has not been published on PyPi yet, it CANNOT be install using pip.

For now, the suggested method is to put the file `register.py` in the same directory as your source files and call ``from register import *``.

Load and register images
**************************************************
.. code-block:: python


    IF_image = imread('IF_image_path')
    IMC_image = imread('IMC_image_path')


    # preprocess IMC image
    IMC_image = preprocess_IMC_nuclear(IMC_image)

    # preprocess IF image
    IF_image = preprocess_IF_nuclear(IF_image)
    IF_image = approx_scale(IF_image, IMC_image)

    # Get registration matrix
    IF_aligned, M = register(IF_image, IMC_image)

    plot_registration(IF_aligned, IMC_image)
    plt.show()