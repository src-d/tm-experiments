# Topic Modeling Experiments on Source Code

## Getting Started

_In the following we do not do so, however it is good practice to limit the amount of memory docker containers have access to with the `-m` flag._

Start by cloning the repository, then building the docker image:

```
git clone https://github.com/src-d/tm-experiments
```

You should also create a directory for all the data. In the following, we call this `/path/to/data`, and will be using defaults values provided by the container for inputs/outputs.

### Preprocess

Launch Babelfish, install all drivers and check you can see them:

```
docker run -d --rm --name tmexp_bblfshd --privileged -p 9432:9432 -v /var/lib/bblfshd:/var/lib/bblfshd bblfsh/bblfshd:latest-drivers
docker exec -it tmexp_bblfshd bblfshctl driver install --all -f
docker exec -it tmexp_bblfshd bblfshctl driver list
```

Clone one or multiple repository in a directory, then launch gitbase:

```
docker run -d --rm --name gitbase -p 3306:3306 --link tmexp_bblfshd:tmexp_bblfshd -e BBLFSH_ENDPOINT=tmexp_bblfshd:9432 -v /path/to/repos:/opt/repos srcd/gitbase:latest
```

Finally, launch the preprocessing (we give the docker socket in order to be able to relaunch Babelfish, however this is a temporary hack):

```
docker run --rm -i -v /var/run/docker.sock:/var/run/docker.sock -v /path/to/data:/data --link tmexp_bblfshd:tmexp_bblfshd --link gitbase:gitbase tmexp preprocess -r repo
```

Once your job is finished, the output file should be located in `/path/to/data/features/`. Unless you wish to run this command once more, you can remove the Babelfish and Gitbase containers, as they will not be of further use.

### Create BoW

You can launch the bag-of-words creation with:

```
docker run --rm -i -v /path/to/data:/data tmexp create_bow --topic-model diff --dataset-name dataset
```

Once your job is finished, the output files should be located in `/path/to/data/bow/dataset/`

### Train HDP

You can launch the training with:

```
docker run --rm -i -v /path/to/data:/data tmexp train_hdp --dataset-name dataset
```

Once your job is finished, the output files should be located in `/path/to/data/topics/dataset/experiment_1`. If you want something more verbose, you can also specify the name of your experiement.

