#!/bin/bash

rm submission.zip
zip -r submission.zip 'data/tiny-imagenet-200/wnids.txt' 'report/report.pdf' 'README.txt' 'requirements.txt' 'test_submission.py' 'model.py' 'params/denoise-final.pt' 'params/mobilenet-final.pt'
