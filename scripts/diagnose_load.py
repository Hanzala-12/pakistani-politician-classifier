import sys
sys.path.insert(0, '.')
from backend import model_loader
import torch
path='project_outputs/models/resnet50_best.pth'
ck=torch.load(path,map_location='cpu')
state=ck['model_state_dict']
print('Sample checkpoint keys:')
for k in list(state.keys())[:40]:
    print(k)
model = model_loader.build_classifier_model('resnet50', num_classes=len(ck['class_names']))
print('\nModel.fc type:', type(model.fc))
print('Sample model state dict keys:')
for k in list(model.state_dict().keys())[:40]:
    print(k)
try:
    model.load_state_dict(state)
    print('\nLoaded successfully')
except Exception as e:
    print('\nLoad error:')
    print(e)
    ck_keys=[k for k in state.keys() if 'fc' in k]
    print('\nCheckpoint fc-like keys:', ck_keys[:40])
    mdl_keys=[k for k in model.state_dict().keys() if 'fc' in k]
    print('Model fc-like keys:',mdl_keys)
