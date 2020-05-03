import os

_home_dir = os.path.expanduser('~') + '/'
_project_rel = 'AnacondaProjects/ilsvrc/'
_project_dir = _home_dir + _project_rel

dataset = _project_dir + 'imagenet/'
training = dataset + 'Training/'
validation = dataset + 'Validation/'

constants = _project_dir + 'shared/constants/'

models = _project_dir + 'models/'
shared = _project_dir + 'shared/'
