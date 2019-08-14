# Topic Modeling Experiments on Source Code

## Getting Started

_In the following we do not do so, however, it is good practice to limit the amount of memory docker containers have access to with the `-m` flag._

Start by cloning the repository, and building the docker image:

```
git clone https://github.com/src-d/tm-experiments
docker build tm-experiments -t tmexp
```

If you have GPU(s) and want to use our lit [Neural Identifier Splitter](https://arxiv.org/abs/1805.11651) then consider building the image with:

```
docker build tm-experiments -t tmexp --build-arg USE_NN=true
```

In all of the following, the created data will be stored in a single directory (hereinafter referred to as `/path/to/data`), which would have the following structure if you were to run each command with the exact same arguments:

```
data
├── datasets
│   ├── my-dataset.pkl
│   ├── my-dataset-2.pkl
│   └── my-merged-dataset.pkl
├── bows
│   └── my-bow
│       ├── doc.bow_tm.txt
│       ├── docword.bow_tm.txt
│       ├── refs.bow_tm.txt
│       └── vocab.bow_tm.txt
└── topics
    └── my-bow
        ├── my-artm-exp
        │   ├── doctopic.npy
        │   ├── wordtopic.npy
        │   └── labels.txt
        └── my-hdp-exp
            ├── doctopic.npy
            └── wordtopic.npy
```

For each commands we only specify the required arguments, check the optional ones with `docker run --rm -i tmexp $CMD --help`.

### `preprocess` command

This command will create a dataset from a cloned git repository. Before launching the command, you will thus need to clone one (or multiple) repository in a directory, as well as start the [Babelfish](https://doc.bblf.sh/) and [Gitbase](https://docs.sourced.tech/gitbase/) servers:

```
make start REPOS=~/path/to/cloned/repos
```

You can now launch the preprocessing with the following command _(we give the docker socket in order to be able to relaunch Babelfish, however this is a **temporary** hack)_:

```
docker run --rm -it -v /path/to/data:/data \ 
                    -v /var/run/docker.sock:/var/run/docker.sock \
                    --link tmexp_bblfshd:tmexp_bblfshd \
                    --link tmexp_gitbase:tmexp_gitbase \
  tmexp preprocess -r repo --dataset-name my-dataset
```

Once this job is finished, the output file should be located in `/path/to/data/datasets/`. Unless you wish to run this command once more, you can remove the Babelfish and Gitbase containers, as they will not be of further use, with:

`docker stop tmexp_gitbase tmexp_bblfshd`

For the sake of explaining the next command, we assume you ran it a second time, and created a dataset named `my-dataset-2`.

### `merge` command

This command will merge multiple dataset created by the previous command. Assuming you created the two datasets, `my-dataset` and `my-dataset-2`, you can launch the merging with the following command:

```
docker run --rm -it -v /path/to/data:/data \ 
  tmexp merge -i my-dataset my-dataset-2 -r repo --dataset-name my-merged-dataset
```

Once this job is finished, the output file should be located in `/path/to/data/datasets/`.

### `create-bow` command

This command will create the input used for the topic modeling, ie bags of words, from datasets created by one of the above commands. You will need to choose between one of two topic evolution models:
- `hall`: each blob is considered to be a document 
- `diff`: we create _delta-documents_ of added and deleted words for each series of documents in the hall model, using the tagged reference order for each repository

For more information about these evolutions models, you can check out [this paper](https://arxiv.org/abs/1704.00135).

You can launch the bag of words creation with the following command (don't forget to specify the dataset name and the topic evolution model you want to use):

```
docker run --rm -it -v /path/to/data:/data \
  tmexp create-bow --topic-model diff --dataset-name my-dataset --bow-name my-bow
```

Once this job is finished, the output files should be located in `/path/to/data/bows/my-bow/`

### `train-artm` command

This command will create an ARTM model from the bags-of-words created previously. You will need to specify the minimum amount of documents belonging to a given topic (by default, belonging means the document has a topic probability over .5) for this topic to be kept, which can either be an absolute number of documents, or relative to the number of documents. For more information about ARTM models, you can check out [this paper](https://link.springer.com/article/10.1007/s10994-014-5476-6) or the [BigARTM documentation](http://docs.bigartm.org/en/stable/index.html).

You can launch the training with the following command (don't forget to specify the bow name and one of the `min-docs` arguments):

```
docker run --rm -it -v /path/to/data:/data \
  tmexp train-artm --bow-name my-bow --exp-name my-artm-exp --min-docs-abs 1
```

Once this job is finished, the output files should be located in `/path/to/data/topics/my-bow/my-artm-exp`.

### `train-hdp`command

This command will allow you to create an HDP model from the bags-of-words created previously. For more information about HDP models, [this paper](https://people.eecs.berkeley.edu/~jordan/papers/hdp.pdf) or the [Gensim documentation](https://radimrehurek.com/gensim/models/hdpmodel.html).

You can launch the training with the following command (don't forget to specify the bow name):

```
docker run --rm -it -v /path/to/data:/data \
  tmexp train-hdp --bow-name my-bow --exp-name my-hdp-exp
```

Once this job is finished, the output files should be located in `/path/to/data/topics/my-bow/my-hdp-exp`.

### `label` command

This command will automatically label topics of a previously created model. You will need to choose between one of the following context creation methods, used to computed word probabilities:
- `hall`: the context will be the hall model of the corpus, ie each blob will be a document in the context
- `last`: blobs from only the last reference of each repo will be a document in the context
- `mean`/`median`/`max`: the mean, max or median word count of each document (from the hall model) across all refs where it exists will be taken as a document in the context
- `concat`: the concatenation of all **added** delta-documents (from the diff model) of the corpus will be a document in the context

For more information about the method used to label topics, you can check out [this paper](https://arxiv.org/abs/1704.00135).

You can launch the labeling with the following command (don't forget to specify the bow name, experience name and one the context creation method):

```
docker run --rm -it -v /path/to/data:/data \
  tmexp label --bow-name my-bow --exp-name my-artm-exp --context hall
```

Once this job is finished, the output file should be located in `/path/to/data/topics/my-bow/my-artm-exp`.
