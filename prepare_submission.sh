#!/bin/bash

rm submission.zip
zip -r submission.zip 'report/report.pdf' 'README.txt' 'requirements.txt' 'test_submission.py' 'model.py'
