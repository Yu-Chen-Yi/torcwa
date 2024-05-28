# Import
import numpy as np
import random
import torch
import scipy.io
from matplotlib import pyplot as plt
import time
import torch.optim as optim
import torcwa
import Materials
import argparse
import os
import cv2
#===== Training settings =====#
parser = argparse.ArgumentParser(description='Diffraction optical elememt - RCWA - Inversed design')
parser.add_argument('--sim_dtype', default=torch.complex64, help='')
parser.add_argument('--geo_dtype', default=torch.float32, help='')
parser.add_argument('--cuda',default=True, action='store_true', help='use cuda?')
parser.add_argument('--wavelength', type=float, default=940., help='Wavelength of simulation. Default=940.(nm)')
parser.add_argument('--seed', type=int, default=777, help='random seed to use. Default=777')
parser.add_argument('--x_period', type=float, default=4041., help='x direction of Period. Default=4041.(nm)')
parser.add_argument('--y_period', type=float, default=15297., help='y direction of Period. Default=15297.(nm)')
parser.add_argument('--dx', type=float, default=4., help='Grid size for xaxis. Default=4.(nm)')
parser.add_argument('--dy', type=float, default=4., help='Grid size for yaxis. Default=4.(nm)')
parser.add_argument('--thickness', type=float, default=235., help='thickness of DOE. Default=235.(nm)')
parser.add_argument('--x_order', type=int, default=13, help='Fourier x order. Default=13')
parser.add_argument('--y_order', type=int, default=25, help='Fourier y order. Default=25')
parser.add_argument('--blur_radius', type=float, default=150., help='blur kernel for critical dimension of DOE. Default=150.(nm)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--printEvery', type=int, default=30, help='number of batches to print average loss')
parser.add_argument('--nEpochs', type=int, default=800, help='number of epochs for training')
args = parser.parse_args()



def get_order(sim,orders=[0,0],direction='forward',port='transmission',ref_order=[0,0]):
    '''
        Return T_all.

        Parameters
        - orders: selected orders (Recommended shape: Nx2)
        - direction: set the direction of light propagation ('f', 'forward' / 'b', 'backward')
        - port: set the direction of light propagation ('t', 'transmission' / 'r', 'reflection')
        - polarization: set the input and output polarization of light ((output,input) xy-pol: 'xx' / 'yx' / 'xy' / 'yy' , ps-pol: 'pp' / 'sp' / 'ps' / 'ss' )
        - ref_order: set the reference for calculating S-parameters (Recommended shape: Nx2)
        - power_norm: if set as True, the absolute square of S-parameters are corresponds to the ratio of power
        - evanescent: Criteria for judging the evanescent field. If power_norm=True and real(kz_norm)/imag(kz_norm) < evanscent, function returns 0 (default = 1e-3)

        Return
        - T_all (torch.Tensor)
    '''
    txx = sim.S_parameters(orders=orders,direction=direction,port=port,polarization='xx',ref_order=ref_order)
    tyy = sim.S_parameters(orders=orders,direction=direction,port=port,polarization='yy',ref_order=ref_order)
    txy = sim.S_parameters(orders=orders,direction=direction,port=port,polarization='xy',ref_order=ref_order)
    tyx = sim.S_parameters(orders=orders,direction=direction,port=port,polarization='yx',ref_order=ref_order)
    T_all = torch.abs(txx)**2 + torch.abs(tyy)**2 + torch.abs(txy)**2 + torch.abs(tyx)**2
    return T_all

def objective_function(rho):
    harmonic_order = [args.y_order, args.x_order]

    
    sim = torcwa.rcwa(freq=1/lamb0,order=harmonic_order,L=L,dtype=sim_dtype,device=device)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
    layer0_eps = rho*silicon_eps + (1.-rho)
    sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
    sim.solve_global_smatrix()

    T_hex1 = [get_order(sim,orders=[xorder,yorder],direction='forward',port='transmission',ref_order=[0,0]) for xorder in range(-12,13,2) for yorder in range(-2,3,2)]
    T_hex2 = [get_order(sim,orders=[xorder,yorder],direction='forward',port='transmission',ref_order=[0,0]) for xorder in range(-11,12,2) for yorder in range(-3,4,2)]
    T_hex = torch.cat([*T_hex1, *T_hex2])
    # Calculate the minimum and maximum of the concatenated tensor
    min_T = torch.min(T_hex)
    max_T = torch.max(T_hex)
    efficiency = torch.sum(T_hex)
    # Compute the uniformity value
    uniformity = (min_T / max_T)

    FoM = efficiency + uniformity
    return FoM

def hexagonal_sym(rho):
    height, width = rho.shape
    rho_left_top = rho[:height//2,:width//2]
    sub_height, sub_width = rho_left_top.shape
    # Create a mask with Boolean diagonal (vectorized operation)
    x, y = torch.meshgrid(torch.arange(sub_height, device=device), torch.arange(sub_width, device=device), indexing='ij')
    mask = y*sub_height/sub_width < x

    # Apply mask to rho_left_top, set elements that fulfill to 0
    rho_left_top[mask] = 0
    rho_left_top = rho_left_top + torch.fliplr(torch.flipud(rho_left_top))
    
    # Creat a rho_right_top is symmertic to rho_left_top
    rho_right_top = torch.fliplr(rho_left_top)

    # Concatenate the top triangle and bottom triangle
    rho_top = torch.concatenate((rho_left_top, rho_right_top), axis=1)

    # Creat a rho_bottom is symmertic to rho_top
    rho_bottom = torch.flipud(rho_top)

    # Concatenate the top rectangle and bottom rectangle
    rho_hex = torch.concatenate((rho_top, rho_bottom), axis=0)
    return rho_hex

def rectangle_sym(rho):
    # Creat a rho_bottom is symmertic to rho_top
    rho_rect = (rho + torch.fliplr(rho))/2
    rho_rect = (rho_rect + torch.flipud(rho_rect))/2
    return rho_rect

def blur_kernel(blur_radius=150.):
    # Blur kernel
    #blur_radius = 150.0
    dx, dy = L[0] / torcwa.rcwa_geo.nx, L[1] / torcwa.rcwa_geo.ny
    x_kernel_axis = (torch.arange(torcwa.rcwa_geo.nx, dtype=geo_dtype, device=device) - (torcwa.rcwa_geo.nx - 1) / 2) * dx
    y_kernel_axis = (torch.arange(torcwa.rcwa_geo.ny, dtype=geo_dtype, device=device) - (torcwa.rcwa_geo.ny - 1) / 2) * dy
    x_kernel_grid, y_kernel_grid = torch.meshgrid(x_kernel_axis, y_kernel_axis, indexing='ij')
    g = torch.exp(-(x_kernel_grid ** 2 + y_kernel_grid ** 2) / blur_radius ** 2)
    g = g / torch.sum(g)
    g_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g)))
    return g_fft

def train(f, rho, beta, epoch):

    epoch_loss = 0
    
    gar_initial = 0.03

    #gar = gar_initial * 0.5 * (1 + np.cos(np.arange(start=0, stop=args.nEpochs) * np.pi / args.nEpochs))


    # Initialize optimizer
    rho.requires_grad_(True)
    print(f'rho1 = {rho}')
    optimizer = optim.Adam([rho], lr=gar_initial, betas=(0.9, 0.999), eps=1.e-8)
    optimizer.zero_grad()
    print(f'rho1 + grad1*lr1 = {rho}')
    # rho conv g(blur kernel) = ifft(rho_fft * g_fft)
    rho_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
    rho_bar = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(rho_fft * g_fft))))

    # per project to 0-1
    rho_tilda = 1 / 2 + torch.tanh(2 * beta * rho_bar - beta) / (2 * torch.tanh(beta))
    print(f'rho_tilda = {rho_tilda}')
    FoM = objective_function(rho_tilda)
    epoch_loss += FoM.data
    # auto gradiant for d_L/d_rho * d_rho/d_rho_i (chain rule)
    FoM.backward()

    with torch.no_grad():
        # Empoly the Adam operimzer to update the new weight rho += Adam[lr] * grad
        optimizer.step()
        # Clamping rho values rho[rho>1] = 1 & rho[rho<0] = 0
        rho.clamp(0, 1)
        rho = hexagonal_sym(rho)

        if (epoch+1)%args.printEvery == 0:
            print("===> Epoch[{}]: Avg. Loss: {:.4f}".format(epoch, epoch_loss/args.printEvery))
            f.write("===> Epoch[{}]): Avg. Loss: {:.4f}\n".format(epoch, epoch_loss/args.printEvery))
            epoch_loss = 0

def checkpoint(rho, epoch):
    if (epoch+1)%args.printEvery == 0:
        save_name = f'epoch_{epoch}.png'
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path,rho.cpu().numpy()*255)
        print("Checkpoint saved to {}".format(save_path))
        
        save_name = f'epoch_{epoch}.mat'
        save_path = os.path.join(save_dir, save_name)
        save_data = {'rho':rho.cpu().numpy()}
        scipy.io.savemat(save_path,save_data)




if __name__ == '__main__':
    #===== Main procedure =====#
    save_dir = f'./DOE_xorder{args.x_order}_yorder_{args.y_order}_THK{int(args.thickness)}_Lx{int(args.x_period)}_Ly{int(args.y_period)}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Hardware
    # If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
    # If you need accurate operation, you have to disable the flag below.
    torch.backends.cuda.matmul.allow_tf32 = False
    sim_dtype = args.sim_dtype
    geo_dtype = args.geo_dtype

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Simulation environment
    # light
    lamb0 = torch.tensor(args.wavelength,dtype=geo_dtype,device=device)    # nm
    inc_ang = 0.*(np.pi/180)    # radian
    azi_ang = 0.*(np.pi/180)    # radian

    # material
    substrate_eps = 1.46**2
    silicon_eps = Materials.aSiH.apply(lamb0)**2

    # geometry
    L = [args.x_period , args.y_period]            # nm / nm
    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = L[0]
    torcwa.rcwa_geo.Ly = L[1]
    torcwa.rcwa_geo.nx = int(L[0]//args.dx)
    torcwa.rcwa_geo.ny = int(L[1]//args.dy)
    torcwa.rcwa_geo.grid()
    torcwa.rcwa_geo.edge_sharpness = 1000. #????

    x_axis = torcwa.rcwa_geo.x.cpu()
    y_axis = torcwa.rcwa_geo.y.cpu()
    # layers
    layer0_thickness = 235.
    g_fft = blur_kernel(blur_radius=args.blur_radius)

    # Load and preprocess image
    image_path = 'check_point.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (torcwa.rcwa_geo.ny, torcwa.rcwa_geo.nx)) / 255.0
    rho = torch.tensor(image_resized, dtype=geo_dtype, device=device)
    rho = hexagonal_sym(rho)
    print(f'initial = {objective_function(rho)}')
    beta = np.exp(np.arange(start=0, stop=args.nEpochs) * np.log(1000) / args.nEpochs)
    beta = torch.from_numpy(beta)

    save_dir = f'./DOE_xorder{args.x_order}_yorder_{args.y_order}_THK{int(args.thickness)}_Lx{int(args.x_period)}_Ly{int(args.y_period)}'
    save_name = f'DOE_xorder{args.x_order}_yorder_{args.y_order}_THK{int(args.thickness)}_Lx{int(args.x_period)}_Ly{int(args.y_period)}.log'
    save_path = os.path.join(save_dir, save_name)
with open(save_path,'w') as f:
    f.write(f'DOE_xorder{args.x_order}_yorder_{args.y_order}_THK{int(args.thickness)}_Lx{int(args.x_period)}_Ly{int(args.y_period)}\n')
    f.write('dataset configuration: epoch size = {}\n'.format(args.nEpochs))
    print('------------------------------------------------')
    for epoch in range(1, args.nEpochs+1):
        train(f, rho, beta[epoch], epoch)
        #validate(f)
        checkpoint(rho, epoch)