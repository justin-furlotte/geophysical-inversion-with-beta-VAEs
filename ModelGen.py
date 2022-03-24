from SimPEG.utils import model_builder
import numpy as np
from discretize import TensorMesh
import matplotlib.pyplot as plt
import torch

'''
File to generate layered earth models. 
Done Lazily in a loop so slow, but doesn't need to be done often
'''

#Simpeg code to define mesh (Discretization where forward modeling and inversions are ran)
dhx = 10 # base cell width
dhz = 10 # base cell depth
nbcx = 65 # num. base cells x
nbcz = 65 # num. base cells z

# Define the base mesh
hx = [(dhx, nbcx)]
hz = [(dhz, nbcz)]
mesh = TensorMesh([hx,hz], x0="CN")
meshcc = mesh.cell_centers
model = np.zeros(mesh.nC)

#max and min of x and z, used later on
zmin = min(meshcc[:,1])
zmax = max(meshcc[:,1])
xmin = min(meshcc[:,0])
xmax = max(meshcc[:,0])

#List to add models
model_list = []

for i in range(100000):
    model = np.ones(mesh.nC)

    # Add wavy layer
    #Randomly sample geological statistics from normal distribution
    layer_loc_1 = np.random.normal(-100,15,1)[0]
    layer_loc_2 = np.random.normal(-180, 15, 1)[0]
    layer_loc_3 = np.random.normal(-250, 15, 1)[0]
    amp_wave = np.random.normal(0,30,1)[0]
    cycles_wave = np.random.normal(1.4,.8,1)[0]
    decay_loc = np.random.randint(-300,300)
    decay = np.random.randint(100,700)

    ra2 = np.random.rand(1)

    ind_layer_1 = meshcc[:, 1] > (layer_loc_1 - amp_wave * np.exp(-np.abs((meshcc[:, 0]+decay_loc)/ decay))*np.cos(
        (meshcc[:, 0] * 2 * np.pi) * cycles_wave / (xmax - xmin) + ra2 * 2 * np.pi))
    ind_layer_2 = meshcc[:, 1] > (
                layer_loc_2 - amp_wave * np.exp(-np.abs((meshcc[:, 0] + decay_loc) / decay) ) * np.cos(
            (meshcc[:, 0] * 2 * np.pi) * cycles_wave / (xmax - xmin) + ra2 * 2 * np.pi))
    ind_layer_3 = meshcc[:, 1] > (
                layer_loc_3 - amp_wave * np.exp(-np.abs((meshcc[:, 0] + decay_loc) / decay) ) * np.cos(
            (meshcc[:, 0] * 2 * np.pi) * cycles_wave / (xmax - xmin) + ra2 * 2 * np.pi))

    model[ind_layer_3] = 1
    model[ind_layer_2] = 2
    model[ind_layer_1] = 3

    ind_conductor = model_builder.getIndicesSphere(np.r_[-0, -250.0], 80.0, meshcc)

    #Adding conductor for tests
    #model[ind_conductor] = 4

    #Set densities for layers
    ld = np.random.normal(.2,.01,1)
    if ld<0:
        ld = 0
    mld = np.random.normal(.5, .01, 1)
    mhd = np.random.normal(.8, .01, 1)
    hd = -1

    if mhd>1.0:
        mhd=1.0



    grav_model = np.copy(model)
    grav_model[model==1] = mhd
    grav_model[model == 2] = mld
    grav_model[model == 3] = ld
    grav_model[model == 4] = hd

    #Turn on to run single example to test inversion
    #np.save('VAEGrav_le',grav_model)

    #Reformat to put in torch consistent
    reshape_grav_model = np.reshape(grav_model, [nbcx, nbcz])
    channel_list = []
    channel_list.append(reshape_grav_model)
    model_list.append(channel_list)


    #Uncomment to plot
    '''
    plt.imshow(reshape_grav_model,origin='lower')
    plt.colorbar()
    plt.show()
    '''

#Save geology as pytorch file
iocg_torch = torch.Tensor(model_list)
torch.save(iocg_torch, 'simple_layers_test.pt')