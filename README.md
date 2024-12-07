Adaptive Arithmetic Prototype
===============================

A prototype site for delivering adaptive arithmetic questions.

It's basically a toy example for me to test and learn about Reinforcement Learning (and other) models.

How it works
-------------

* Reinforcement Learning (RL) has been used to train a model
* The user first gets a pre-test to determine their current ability in addition, subtraction, multiplication and 
  division
* The model then delivers new questions, based on the users pre-test knowledge and then subsequent responses to the new 
  questions, to give progressively more difficult questions
* 

Well, that's how it's supposed to work!

Installation
--------------

The site uses Django, so the installation should be the same as for any other Django site:

See: [Installation](docs/install.md)

Models and training
----------------------
All the code for configuring and retraining the model(s) is provided.

See: [Models and training](docs/training.md)


