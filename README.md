# Visual Acoustic Matching for Dubbing (VAMDub)

We use [Visual Acoustic Matching](https://github.com/facebookresearch/visual-acoustic-matching) and [SeamlessM4Tv2](https://github.com/facebookresearch/seamless_communication) to create an end-to-end film dubbing system.


## Film Dubbing

Generally, during the process of film dubbing, there is access to dialogues cleanly recorded by the actors in studio environment, and the scene in which the dialogue is to be put. The dialogue is first recorded in target language by a voice actor, and then the effects of the environment are added to the dialogue.


We identify the key requirements of an automatic dubbing system as follows:
- Preserve the speaker's identity and prosidic features while translating from source language to target language.
- Automatically add the environmental effects (acoustics) to the translated dialogue given the recorded scene (either as video or image).


## Steps to run the project

Before running the project, you'll have to download the [pretrained weights](http://dl.fbaipublicfiles.com/vam/pretrained-models.zip) and unzip it under the `checkpoints/vam` subdirectory.

```sh
$ wget http://dl.fbaipublicfiles.com/vam/pretrained-models.zip
```

Create a docker container, which sets up the environmnt required to run the project:
```sh
$ docker build -t vam-dub .
```

Once the container is created, you can launch it as follows:
```sh
$ sh scripts/launch_container.sh
```

The server can which makes the inference has to launched before running the demo:
```sh
$ python demo/api.py
```

The demo can then be launched using the following two command:
```sh
$ streamlit demo/demo.py
```