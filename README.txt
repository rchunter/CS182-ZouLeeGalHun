CS 182 Final Project: Vision Project (Generalizable Classifiers)
Rowen Hunter, Eero Gallano, Jonathan Lee
----------------------------------------

Project repo: https://github.com/rchunter/CS182-ZouLeeGalHun

Project contents:
  * `README.txt` - This file
  * `requirements.txt` - Python package dependencies
  * `model.py` - PyTorch model definitions
  * `test_submission.py` - Submission script to classify all the images listed in `eval.csv`
  * `data/tiny-imagenet-200/wnids.txt` - List of class names that `test_submission.py` will read. You can also get this file from the `tiny-imagenet-200` dataset (http://cs231n.stanford.edu/tiny-imagenet-200.zip).
  * `report/report.pdf` - Our project report
  * `params/*-final.pt` - Final model parameters

Notes
-----
  * You should run `test_submission.py` with Python 3 (preferably Python 3.8).
  * Unfortunately, `test_submission.py` can be quite slow because the files have to be read one at a time and classified one-by-one, instead of in minibatches. Using a GPU may help speed up the forward pass.
  * We assume the images are RGB, since the number of channels our model expects (3) is fixed.
