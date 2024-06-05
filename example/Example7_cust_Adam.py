# Import
import numpy as np
import random
import torch
import scipy.io
from matplotlib import pyplot as plt
import time
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
parser.add_argument('--seed', type=int, default=333, help='random seed to use. Default=777')
parser.add_argument('--x_period', type=float, default=4041., help='x direction of Period. Default=4041.(nm)')
parser.add_argument('--y_period', type=float, default=15297., help='y direction of Period. Default=15297.(nm)')
parser.add_argument('--dx', type=float, default=4., help='Grid size for xaxis. Default=4.(nm)')
parser.add_argument('--dy', type=float, default=4., help='Grid size for yaxis. Default=4.(nm)')
parser.add_argument('--thickness', type=float, default=235., help='thickness of DOE. Default=235.(nm)')
parser.add_argument('--x_order', type=int, default=9, help='Fourier x order. Default=13')
parser.add_argument('--y_order', type=int, default=27, help='Fourier y order. Default=25')
parser.add_argument('--blur_radius', type=float, default=150., help='blur kernel for critical dimension of DOE. Default=150.(nm)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1. Default=0.9')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2. Default=0.999')
parser.add_argument('--epsilon', type=float, default=1.e-8, help='Adam epsilon. Default=1.e-8')
parser.add_argument('--printEvery', type=int, default=30, help='number of batches to print average loss')
parser.add_argument('--nEpochs', type=int, default=800, help='number of epochs for training')
args = parser.parse_args()

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
# Check GPU deive was activated
if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Fix the random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


# Simulation environment
# light source setting
lamb0 = torch.tensor(args.wavelength,dtype=geo_dtype,device=device)    # nm
inc_ang = 0.*(np.pi/180)    # radian
azi_ang = 0.*(np.pi/180)    # radian

# material setting
substrate_eps = 1.46**2
silicon_eps = Materials.aSiH.apply(lamb0)**2

# geometry setting
L = [args.y_period , args.x_period]            # nm / nm
torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = L[0]
torcwa.rcwa_geo.Ly = L[1]
torcwa.rcwa_geo.nx = int(L[0]//args.dy)
torcwa.rcwa_geo.ny = int(L[1]//args.dx)
torcwa.rcwa_geo.grid()
torcwa.rcwa_geo.edge_sharpness = 1000. #????

# layers
layer0_thickness = args.thickness

def get_exist_order():
    # order
    n = np.arange(-args.x_order//2, args.x_order//2 +1)
    m = np.arange(-args.y_order//2, args.y_order//2 +1)
    yorder, xorder = np.meshgrid(m, n)
    kx = xorder * lamb0.cpu().numpy()/L[1]
    ky = yorder * lamb0.cpu().numpy()/L[0]
    exist_xorder = xorder[np.where((kx**2 + ky**2) <= 1)]
    exist_yorder = yorder[np.where((kx**2 + ky**2) <= 1)]
    return exist_xorder, exist_yorder

def hexagonal_sym(rho):
    height, width = rho.shape
    rho_left_top = rho[:height//2,:width//2]
    sub_height, sub_width = rho_left_top.shape
    # 創建布爾對角 (向量化操作)
    x, y = torch.meshgrid(torch.arange(sub_height, device=device), torch.arange(sub_width, device=device), indexing='ij')
    mask = y*sub_height/sub_width < x

    # 將掩膜應用於rho_left_up，將滿足條件的元素設置為0
    rho_left_top[mask] = 0
    rho_left_top = rho_left_top + torch.fliplr(torch.flipud(rho_left_top))

    rho_right_top = torch.fliplr(rho_left_top)

    # 拼接整個矩陣 (左右部分)
    rho_top = torch.concatenate((rho_left_top, rho_right_top), axis=1)

    rho_bottom = torch.flipud(rho_top)
    # 拼接整個矩陣 (上下部分)
    rho_hex = torch.concatenate((rho_top, rho_bottom), axis=0)
    return rho_hex

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

    T_all = [get_order(sim,orders=[xorder,yorder],direction='forward',port='transmission',ref_order=[0,0]) for xorder, yorder in zip(exist_yorder, exist_xorder)]
    T_all = torch.cat(T_all)
    # Calculate the minimum and maximum of the concatenated tensor
    min_T = torch.min(T_hex)
    max_T = torch.max(T_hex)
    efficiency = torch.sum(T_hex) / torch.sum(T_all)
    # Compute the uniformity value
    uniformity = min_T / max_T

    FoM = efficiency + uniformity
    return FoM

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

def blur_process(rho):
    # rho conv g(blur kernel) = ifft(rho_fft * g_fft)
    rho_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
    rho_bar = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(rho_fft * g_fft))))
    return rho_bar

def project_process(rho_bar,beta):
    rho_tilda = 1 / 2 + torch.tanh(2 * beta * rho_bar - beta) / (2 * torch.tanh(beta))
    return rho_tilda

def validate(rho,f):
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

    T_all = [get_order(sim,orders=[xorder,yorder],direction='forward',port='transmission',ref_order=[0,0]) for xorder, yorder in zip(exist_yorder, exist_xorder)]
    T_all = torch.cat(T_all)
    # Calculate the minimum and maximum of the concatenated tensor
    min_T = torch.min(T_hex)
    max_T = torch.max(T_hex)
    efficiency = torch.sum(T_hex) / torch.sum(T_all)
    # Compute the uniformity value
    uniformity = min_T / max_T
    print("===> Efficiency: {:.2f} %".format(efficiency*100))
    print("===> Uniformity: {:.2f} %".format(uniformity*100))
    f.write("===> Efficiency: {:.2f} %\n".format(efficiency*100))
    f.write("===> Uniformity: {:.2f} %\n".format(uniformity*100))

exist_xorder, exist_yorder = get_exist_order()

# Perform optimization
beta = torch.exp(torch.arange(100, args.nEpochs+100) * torch.log(torch.tensor(1000.0)) / args.nEpochs)
gar = args.lr * 0.5*(1+np.cos(np.arange(start=0,stop=args.nEpochs)*np.pi/args.nEpochs))
g_fft = blur_kernel(blur_radius=args.blur_radius)

check_point_path = 'check_point_hex3.png'
check_point = cv2.imread(check_point_path, cv2.IMREAD_GRAYSCALE)
check_point = cv2.resize(check_point, (torcwa.rcwa_geo.ny, torcwa.rcwa_geo.nx))/ 255.0 
# Convert image to tensor
rho = torch.tensor(check_point, dtype=geo_dtype, device=device)
#rho = hexagonal_sym(rho)

momentum = torch.zeros_like(rho)
velocity = torch.zeros_like(rho)
FoM_history = []
save_dir = f'./DOE_xorder{args.x_order}_yorder_{args.y_order}_THK{int(args.thickness)}_Lx{int(args.x_period)}_Ly{int(args.y_period)}'
save_name = f'DOE_xorder{args.x_order}_yorder_{args.y_order}_THK{int(args.thickness)}_Lx{int(args.x_period)}_Ly{int(args.y_period)}.log'
save_path = os.path.join(save_dir, save_name)
with open(save_path,'w') as f:
    f.write(f'DOE_xorder{args.x_order}_yorder_{args.y_order}_THK{int(args.thickness)}_Lx{int(args.x_period)}_Ly{int(args.y_period)}\n')
    f.write('dataset configuration: epoch size = {}\n'.format(args.nEpochs))
    print('------------------------------------------------')
    validate(rho,f)
    start_time = time.time()
    for epoch in range(0,args.nEpochs):
        rho.requires_grad_(True)
        rho_bar = blur_process(rho)
        rho_tilda = project_process(rho_bar,beta[epoch])
        FoM = objective_function(rho_tilda)
        FoM.backward()

        with torch.no_grad():
            rho_gradient = rho.grad
            rho.grad = None

            FoM = float(FoM.detach().cpu().numpy())
            FoM_history.append(FoM)

            momentum = (args.beta1*momentum + (1-args.beta1)*rho_gradient)
            velocity = (args.beta2*velocity + (1-args.beta2)*(rho_gradient**2))

            momentum_hat = (momentum / (1-args.beta1**(epoch+1)))
            velocity_hat = (velocity / (1-args.beta2**(epoch+1)))
            
            rho += gar[epoch]* momentum_hat/ (torch.sqrt(velocity_hat) + args.epsilon)
            
            # Clamping rho values rho[rho>1] = 1 & rho[rho<0] = 0
            rho.clamp(0, 1)
            
            rho = hexagonal_sym(rho)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("===> Epoch[{}/{}]: Avg. FoM: {:.5f} Elasped time: {:.1f} s".format(epoch+1, args.nEpochs, FoM, elapsed_time))
            if (epoch+1)%args.printEvery == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                #print("===> Epoch[{}/{}]: Avg. FoM: {:.5f} Elasped time: {:.1f} s".format(epoch+1, args.nEpochs, loss, elapsed_time))
                f.write("===> Epoch[{}/{}]): Avg. FoM: {:.5f}  Elasped time: {:.1f} s\n".format((epoch+1), args.nEpochs, FoM, elapsed_time))
                save_name = f'epoch_{(epoch+1)}.png'
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path,rho_tilda.detach().cpu().numpy()*255)
                print("Checkpoint saved to {}".format(save_path))
                
                save_name = f'epoch_{(epoch+1)}.mat'
                save_path = os.path.join(save_dir, save_name)
                save_data = {'rho':rho_tilda.detach().cpu().numpy()}
                scipy.io.savemat(save_path,save_data)
                validate(rho_tilda,f)
plt.plot(np.array(FoM_history))
# Export data
save_name = 'FoM.mat'
save_path = os.path.join(save_dir, save_name)
ex7_data = {'FoM_history':FoM_history}
scipy.io.savemat(save_path,ex7_data)