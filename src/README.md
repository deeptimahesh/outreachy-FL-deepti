# Federated Learning

Federated Learning is a new subarea of machine learning where the training process is distributed among many users.
Instead of sharing their data, users only have to provide weight updates to the server.

This repository contains the code for the experiments and simulations related to Federated Learning that I'm running at Mozilla.
More information about the plans for using Federated Learning at Mozilla can be found on [Bugzilla](https://bugzilla.mozilla.org/show_bug.cgi?id=1462102).
The results of these experiments are going to be the basis for the experimental part of my master's thesis.

## Components

- `data`: Functions for loading or generating datasets
- `utils`: Generic helper functions or classes
- `optimizers`: Classes that decide how to apply an update to a model
- `simulations`: The main part of this repository that connects everything

## Simulations

- frecency ([data generation](https://github.com/florian/federated-learning/blob/master/data/frecency.ipynb) / [optimization](https://github.com/florian/federated-learning/blob/master/simulations/frecency.ipynb))

## References

- [Blog post](https://florian.github.io/federated-learning/) explaining the concepts behind federated learning
- [Bugzilla](https://bugzilla.mozilla.org/show_bug.cgi?id=1462102)
- [Federated learning addon](https://github.com/florian/federated-learning-addon) (client-side code)
- [Frecency documentation](https://developer.mozilla.org/en-US/docs/Mozilla/Tech/Places/Frecency_algorithm) (a bit outdated)
