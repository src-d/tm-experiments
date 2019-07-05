# Topic Modeling Experiments on Source Code

## Getting Started

_In the following we do not do so, however it is good practice to limit the amount of memory docker containers have access to with the `-m` flag._

Start by cloning the repository, and building the docker image:

```
git clone https://github.com/src-d/tm-experiments
docker build tm-experiments -t tmexp
```

In all of the following, all of the created data will be stored in a single directory (hereinafter referred to as `/path/to/data`), which would have the following structure if you were to run with the exact same names:

```
data
├── datasets
│   └── my-dataset.pkl
├── bows
│   └── my-bow
│       ├── doc.bow_tm.txt
│       ├── docword.bow_tm.txt
│       └── vocab.bow_tm.txt
└── topics
    └── my-bow
        └── my-exp
            ├── doc.topic.txt
            └── word.topic.npy
```

For all commands we only specify the required arguments, check the optional ones with `docker run --rm -i tmexp $CMD --help`

### Preprocess

This command will allow you to create a dataset from a cloned git repository. Before launching the command, you will thus need to clone one (or multiple) repository in a directory, as well as start the Babelfish and Gitbase servers:

```
make start REPOS=~/path/to/cloned/repos
```

You can now launch the preprocessing with the following command _(we give the docker socket in order to be able to relaunch Babelfish, however this is a **temporary** hack)_:

```
docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock -v /path/to/data:/data --link tmexp_bblfshd:tmexp_bblfshd --link tmexp_gitbase:tmexp_gitbase tmexp preprocess -r repo --dataset-name my-dataset
```

Once your job is finished, the output file should be located in `/path/to/data/datasets/`. Unless you wish to run this command once more, you can remove the Babelfish and Gitbase containers, as they will not be of further use, with:

`docker stop tmexp_gitbase tmexp_bblfshd`

### Create BoW

You can launch the bag-of-words creation with the following command (don't forget to specify the dataset name):

```
docker run --rm -it -v /path/to/data:/data tmexp create_bow --topic-model diff --dataset-name my-dataset --bow-name my-bow
```

Once your job is finished, the output files should be located in `/path/to/data/bows/my-bow/`

### Train HDP

You can launch the training with the following command (don't forget to specify the bow name):

```
docker run --rm -it -v /path/to/data:/data tmexp train_hdp --bow-name my-bow --exp-name my-exp
```

Once your job is finished, the output files should be located in `/path/to/data/topics/my-bow/my-exp`.
