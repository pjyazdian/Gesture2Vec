# Gesture2Vec: Clustering Gestures using Representation Learning Methods for Co-speech Gesture Generation

## The Best Paper Award Winner in Cognitive Robotics at IROS2022

This is an official PyTorch implementation of _Gesture2Vec: Clustering Gestures using Representation Learning Methods for Co-speech Gesture Generation_ (IROS 2022). In this paper, we present an automatic gesture generation model that uses a vector-quantized variational autoencoder structure as well as training techniques to learn a rigorous representation of gesture sequences. We then translate input text into a discrete sequence of associated gesture chunks in the learned gesture space. Subjective and objective evaluations confirm the success of our approach in terms of appropriateness, human-likeness, and diversity. We also introduce new objective metrics using the quantized gesture representation.

### [Paper](https://sfumars.com/wp-content/papers/2022_iros_gesture2vec.pdf) | [Demo Video](https://www.youtube.com/watch?v=ac8jWk4fdCU) | [Presentation](https://youtu.be/qFObMpOboCg)

![OVERVIEW](Figures/model.jpg)

## Demo Video

[![Demo Video](https://img.youtube.com/vi/ac8jWk4fdCU/0.jpg)](https://www.youtube.com/watch?v=ac8jWk4fdCU)

## Presentation

[![IROS2022 Presentation](https://img.youtube.com/vi/qFObMpOboCg/0.jpg)](https://www.youtube.com/watch?v=qFObMpOboCg)

## Instructions

To replicate our results as described in our paper, please follow the instructions below.

### Requirements

This project has been tested in the following environment (using a single Nvidia GPU):

-   Ubuntu 20.04.6 LTS
-   Anaconda3 23.9
-   CUDA 12.2

All other dependencies can be found in `gesture2vec.yml` and installed using `conda env create -f gesture2vec.yml`. Activate the environment using `conda activate gesture2vec`. Further details on creating and managing Anaconda environments can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Dataset

The Trinity Speech-Gesture GENEA Challenge 2020 dataset can be found at the [Trinity Speech-Gesture website](https://trinityspeechgesture.scss.tcd.ie/). The entire dataset can be downloaded and saved to a directory named `data` within this project. The resulting directory should look similar to:

```
data
    |
    --train
        |
        --Audio
        --Motion
        --Transcripts
    |
    --valid
        |
        --Audio
        --Motion
        --Transcripts
```

### Data Preprocessing

Run the `trinity_data_to_lmdb` script found in the `scripts` directory. Please refer to the module docstring found at the top of the script for details on how to run the preprocessing script.

### Training

TODO

## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code uses datasets inluding Trinity and Talk With Hand (TWH) that each have their own respective licenses that must also be followed.

Please feel free to contact us (pjomeyaz@sfu.ca) with any question or concerns.
