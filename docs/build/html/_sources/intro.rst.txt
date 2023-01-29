Introduction
============

``smack-imc`` is a lightweight package to register images from IF/IMC. It's designed with a specific input type in mind: images of cells with nuclear staining which have been both stained for immunofluorescence (IF), and processed by imaging mass cytometry (IMC).


Motivation
**********

IF gives us "ground truth" and higher resolution images, IMC gives us proein quantification. To use both, we need to be able to align the images. `smack-imc` packages up some of the hueristics we've learned are useful in this aplication, and minimizes the number of parameters that need to be tuned. 

Limitations
***********

- Minimal - the package does what it says an no more

- Bespoke - `smack-imc` is not flexible. If you like the sound of its functionality, you might be better off looking at the source code and adapting it than trying to use the package directly. 
