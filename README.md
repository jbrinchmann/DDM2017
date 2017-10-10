# DDM 2017
[![Build Status](https://travis-ci.org/jbrinchmann/DDM2017.svg?branch=master)](https://travis-ci.org/jbrinchmann/DDM2017)
## Forking the Repository:
1. Make a github account and log in.
1. Click on the 'Fork' at the top right. This will create a 'fork' on your own account. That means that you now have the latest commit of the repo and its history in your control. If you've tried to 'git push' to the DDM2017 repo you'd have noticed that you don't have access to it.
1. Once it's forked, you can go to your github profile and you'll see a DDM2017 repo. Go to it and get the .git link (green button)
1. Somewhere on your machine, git clone git clone https://github.com/[YOUR_GIT_UNAME]/DDM2017.git. You also need to enter the directory
1. Add our repo as an upstream. That way you can get (pull) new updates: git remote add upstream https://github.com/jbrinchmann/DDM2017.git
1. git remote -v should give:
origin	https://github.com/[YOUR_GIT_UNAME]/DDM2017.git (fetch)
origin	https://github.com/[YOUR_GIT_UNAME]/DDM2017.git (push)
upstream	https://github.com/jbrinchmann/DDM2017.git (fetch)
upstream	https://github.com/jbrinchmann/DDM2017.git (push)
1. Now you're ready to add files and folders to your local fork. Use git add, git commit and git push (origin master) to add your assignments.

