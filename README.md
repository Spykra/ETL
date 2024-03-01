# Brain Tumor Detection Using Deep Learning

## About the Project

This project leverages deep learning techniques to detect, classify, and identify the location of brain tumors from Magnetic Resonance Imaging (MRI) scans. Utilizing Convolutional Neural Networks (CNNs) for multi-task classification, this work aims to contribute to the important research domain of medical imaging by facilitating early detection and efficient treatment planning for brain tumors.

## Methods

The project explores the use of Generative Adversarial Networks (GANs) and diffusion models for generating synthetic brain MRI images, facilitating a broader understanding of tumor characteristics and improving model robustness.

### Models and Training

1. **Basic GAN Model**: Utilizes a generator-discriminator framework to synthesize brain MRI images, aiding in the exploration of tumor features.
2. **Convolutional GAN Model**: Enhances the basic GAN approach by incorporating convolutional layers, enabling the model to better capture spatial hierarchies in the data.
3. **Diffusion Model**: Employs a novel approach to generate high-quality images through a process that gradually denoises input data, offering a promising direction for synthetic data generation.

## Dataset

The dataset used in this project amalgamates three distinct sources: figshare, the SARTAJ dataset, and Br35H, resulting in a total of 7023 human brain MRI images. These images are categorized into four classes: glioma, meningioma, no tumor, and pituitary, with the 'no tumor' class images sourced from the Br35H dataset. Notably, due to categorization issues observed with glioma class images in the SARTAJ dataset, those images were replaced with ones from figshare to ensure data integrity.

The brain tumor MRI dataset used in this project was obtained from Kaggle, courtesy of Msoud Nickparvar. The details of the dataset citation are as follows:

@misc{msoud_nickparvar_2021,
	title={Brain Tumor MRI Dataset},
	url={https://www.kaggle.com/dsv/2645886},
	DOI={10.34740/KAGGLE/DSV/2645886},
	publisher={Kaggle},
	author={Msoud Nickparvar},
	year={2021}
}