
import numpy as np
from discretize import TensorMesh
from SimPEG import maps
from SimPEG.potential_fields import gravity
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
from torch.autograd import grad
from VAE65 import VAE65
from SimPEG import utils
from numpy import inf,random




'''
Define a mesh, or the cells where we solve for the density.
Here is is pseudo 2d, with one y cell extended 650m
'''
dhx = 10 # base cell width
dhy = 650 # single cell width
dhz = 10 #
nbcx = 65 #Number of cells in the x direction
nbcy = 1 #
nbcz = 65 #Number of cells in the -z direction

hx = [(dhx, nbcx)]
hz = [(dhz, nbcz)]
hy = [(dhy,nbcy)]
mesh = TensorMesh([hx,hy,hz], x0="CCN")

meshcc = mesh.cell_centers
nC = mesh.n_cells
ref_model = np.zeros(mesh.nC)
faces_z = mesh.faces_z[:,2]

#Depth weighting for gravity, needed because gravity has poor depth resolution, better lateral resolution
dwz =1/(0-faces_z)**1

dwz[dwz == inf] = 0

ind_x_less= meshcc[:,0]<-350
ind_x_more=meshcc[:,0]>350
ind_x_pad = ind_x_more+ind_x_less

ind_x_core = (
        (meshcc[:,0]<350) &
        (meshcc[:,0]>-350))

#Load different test models
true_geology_g_list = []
true_geology_g = np.load('VAEGrav_LE.npy')
#true_geology_g = np.load('VAEGrav_weaksphere.npy')
#true_geology_g = np.load('VAEGrav_strongsphere.npy')

#Sloppy way to put reference model in form pytorch expects
reshape_grav_model = np.reshape(true_geology_g, [nbcx, nbcz])
ref = np.zeros_like(reshape_grav_model)
ref_list = []
channel_list_ref = []
channel_list = []
channel_list.append(reshape_grav_model)
true_geology_g_list.append(channel_list)
channel_list_ref.append(ref)
ref_list.append(channel_list_ref)

#Define reciever locations and survey. These are the measuments on the surface
num_recievers= 80
x = np.linspace(-400.0, 400.0, num_recievers)
y = np.zeros_like(x)
z = 1*np.ones_like(x)

receiver_locations = np.c_[x, y, z]
components = ["gz"]
receiver_list = gravity.receivers.Point(receiver_locations, components=components)
receiver_list = [receiver_list]
source_field = gravity.sources.SourceField(receiver_list=receiver_list)
survey = gravity.survey.Survey(source_field)

#Forward Simulation, modeling synthetic data so we can use this to solve the inverse problem for the original data

model_map = maps.IdentityMap(nP=nC)
simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    rhoMap=model_map,
)

#Predicted data
dpred = simulation.dpred(true_geology_g)

#Add noise to synthetic data to emulate real conditions
maximum_anomaly = np.max(np.abs(dpred))
noise = 0.001 * maximum_anomaly * np.random.rand(len(dpred))
dpred = dpred+noise

#Get forward modeling matrix from Simpeg
#Triple volume integral with closed form solution Justin wrote on board
G = simulation.get_G()

#Matrices that calculate roughness on the model, in standard inversion, we expect smooth model
Dx = mesh.cellGradx.toarray()
Dz = mesh.cellGradz.toarray()

#Sensitivity weighting matrix
wr = utils.depth_weighting(
            mesh, receiver_locations,
            exponent=2
        )



#This begins pytorch section
#Just convert all variables to torch, annoying
noise = torch.tensor(noise)
tgt= torch.Tensor(true_geology_g_list).type(torch.float)

#Target misfit to hit
#Just want to hit data to within noise, otherwise we are overfitting model to noise
trgtmsft = torch.norm(noise)**2

G = torch.from_numpy(G).type(torch.float)
Dx = torch.from_numpy(Dx).type(torch.float).to_sparse()
Dz = torch.from_numpy(Dz).type(torch.float).to_sparse()
dw = torch.from_numpy(wr).type(torch.float)
dwz = torch.from_numpy(dwz).type(torch.float)
ind_x_pad = torch.from_numpy(ind_x_pad).type(torch.bool)
ind_x_core = torch.from_numpy(ind_x_core).type(torch.bool)

dobs = torch.from_numpy(dpred).type(torch.float)
device = 'cpu'
cuda = False

# Define the latent space dimension:
z_dim = 10

# Settings for multi-CPU or GPU:
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
if str(device) == 'cuda':
    cuda=True
else:
    cuda=False


#Set seed to reproduce results (Git rid of to randomize)
seed=100
torch.manual_seed(seed)
random.seed(seed)

# Load the VAE parameters but not the model:
gpath = 'VAE65_10_a1b10_le_JW.pth'

# Load the VAE model, initialize with parameters
dnnmodel = VAE65(cuda=cuda, gpath=gpath,z_dim=z_dim)
for param in dnnmodel.parameters():
    param.requires_grad = False
dnnmodel.to(device)
dnnmodel.eval() # set evaluation mode (as opposed to training mode).
netG = dnnmodel.decode
netE = dnnmodel.encode

#Define inversion
#Get initial reference model(Set 1.0 to 0 to start with mean model
z_init = 1.0*torch.randn(1,z_dim, dtype=torch.float, device=device)
z=torch.clone(z_init)


image2 = netG(z)

#Plot Modes
x= torch.zeros_like(image2)

z_dim = 10
z_new = 0.0*torch.randn(10,z_dim, dtype=torch.float, device=device)
x_new = netG(z_new)


x1=torch.reshape(netG(z),(-1,))
x1=torch.zeros_like(x1)
x=torch.clone(x1)

d_obs = dpred
d_init = G@(torch.reshape(netG(z),(-1,))*.8-.4)
d_init = d_init.numpy()
x_update= x[ind_x_core]



def Lossx(dobs,G,z,x,lam,lammy,ax,az,aw,ps,px,py,dp):
    #Our loss function

    Gpx = (torch.reshape(netG(z),(-1,))+x) #Latent plus fully dimensional

    dmis = dp*torch.norm(G@Gpx-dobs)**2 #Datamisfit term
    znorm=lam * torch.norm(z) ** 2 #Refularization on latents

    dwx = dw*x #Depth Weighting
    xnorms = lammy*(aw*torch.linalg.norm(dwx,ord=ps)**ps #Regularization of x-component
                  +ax*torch.linalg.norm(dwz*torch.mv(Dx,x),ord=px)**px
                  +az*torch.linalg.norm(dwz*torch.mv(Dz,x),ord=py)**py
                  )

    #Was testing with this earlier, left in
    pad_norm = 50*torch.norm(x[ind_x_pad])**2


    return dmis + znorm + xnorms + pad_norm,dmis



#Parameters for messing around with inversion
#I was testing different stuff, so parts of following are irrelevant
loss_save=100
loss_best = 200
dmis=100
ps=2
px=1
py=1
tr=2
dp=1
numepoch = 3000

#Initial step size for latent, xspace
deltaz = .001
deltax = .001

#Penalizes roughness of x
ax = 1
az = 1
aw = 1

lam = .0001 #Parameter to punish latent vector from deviating from mean
lammy = 10000 #Parameter to punish x

#Save for movie
z_list = []
x_list = []
z_list.append(z_init)

j=0
cnt=0

#Run inversion with line-search

for i in range(numepoch):
    if cnt % 2 == 0:
        deltaz = deltaz*4
    else:
        deltax = deltax*4

    cnt+=1


    z = torch.tensor(z, requires_grad=True)
    x = torch.tensor(x,requires_grad=True)

    loss,dmis = Lossx(dobs,G,z,x,lam=lam,lammy=lammy,ax = ax,az=az,aw=aw,ps=ps,px=px,py=py,dp=dp)


    if cnt%2 == 0:
        gradLoss = grad(loss,z,retain_graph=True)
    else:
        gradxloss = grad(loss, x, retain_graph=True)

    loss = loss.item()
    # update the parameters
    with torch.no_grad():
        j=0

        losstry = loss

        while losstry >= loss:
            j+=1
            if cnt % 2 == 0:
                z_try = z - deltaz*gradLoss[0]
                losstry, dmis_try = Lossx(dobs, G, z_try, x, lam=lam, lammy=lammy, ax=ax, az=az, aw=aw, ps=ps,
                                          px=px, py=py, dp=dp)
                deltaz = deltaz / 2
            else:
                break
                x_try = x - deltax*gradxloss[0]
                losstry, dmis_try = Lossx(dobs, G, z, x_try, lam=lam, lammy=lammy, ax=ax, az=az, aw=aw, ps=ps,
                                          px=px, py=py, dp=dp)
                deltax = deltax / 2

            losstry=losstry.item()

            if j == 5:
                break
        ls = loss
        if cnt % 2 ==0 and losstry<loss:
            z = z_try
            loss = losstry
        if cnt %2 != 0 and losstry<loss:
            x = x_try
            loss = losstry
        if (ls-losstry)/ls>.01:
            cnt+=1



        if i % 100 == 1 and i != 1:
            if (loss_save-loss)/loss_save < .005:
                #Break if inversion is stuck
                print('No Progress')
                break
            loss_save = loss
            print(i, loss,dmis/trgtmsft)

        if i%100 == 1:
            z_list.append(torch.clone(z.detach()))
            x_list.append(torch.clone(torch.reshape(x,[65,65])))

        if dmis/trgtmsft < 1:
            #dmis/trgt is the current data misfit vs our expected, break when ratio is 1
            print(i, loss,dmis/trgtmsft)
            break

#Updating plot of inversion
z=z.detach()
zbest=z.detach()
x = x.detach()
x=x.numpy()

tg = np.reshape(true_geology_g,[65,65])

fig, axs = plt.subplots(nrows=1, ncols=2)

plot = axs[0].imshow(netG(z_init)[0][0]+x_list[0],origin='lower')
plot2 = axs[1].imshow(tg,origin='lower')

def play(f):
    for img in z_list:
        plot.set_data(netG(img)[0][0]+x_list[0])
        plt.draw(), plt.pause(.2)

axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Play')

bnext.on_clicked(play)
plt.show()

