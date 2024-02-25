import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device("cpu")
# A demo outside the Lettuce pipeline
torch.set_default_dtype(torch.float32)
dtype = torch.float32
'''basic configuration'''
N_ITERATIONS = 15000
REYNOLDS_NUMBER = 80

N_POINTS_X = 300
N_POINTS_Y = 50

CYLINDER_CENTER_INDEX_X = N_POINTS_X // 5
CYLINDER_CENTER_INDEX_Y = N_POINTS_Y // 2
CYLINDER_RADIUS_INDICES = N_POINTS_Y // 9

MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04

VISUALIZE = True
PLOT_EVERY_N_STEPS = 100
SKIP_FIRST_N_ITERATIONS = 5000
C_s = 1/np.sqrt(3)

'''Lattice configuration'''
N_DISCRETE_VELOCITIES = 9
LATTICE_VELOCITIES = np.array([
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1,]
])
print(f"D{LATTICE_VELOCITIES.shape[0]}Q{LATTICE_VELOCITIES.shape[1]} lattice")

LATTICE_INDICES = np.array([0,1,2,3,4,5,6,7,8])
OPPOSITE = np.array([0,3,4,1,2,7,8,5,6]) # Opposite velocities which use in bounce back boundary condition; also relate to the indexing of LATICE_VELOCITIES

LATTICE_WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

RIGHT_VELOCITIES = np.array([1, 5, 8])
LEFT_VELOCITIES = np.array([3, 6, 7])
UP_VELOCITIES = np.array([2, 5, 6])
DOWN_VELOCITIES = np.array([4, 7, 8])

PURE_HORIZONTAL_VELOCITIES = np.array([0,1,3])
PURE_VERTICAL_VELOCITIES = np.array([0,2,4])

# transfer to torch tensor
with torch.no_grad():
    LATTICE_VELOCITIES = torch.tensor(LATTICE_VELOCITIES, dtype=dtype,device = device)
    LATTICE_WEIGHTS = torch.tensor(LATTICE_WEIGHTS, dtype=dtype,device = device)
    RIGHT_VELOCITIES = torch.tensor(RIGHT_VELOCITIES, dtype=torch.int,device = device)
    LEFT_VELOCITIES = torch.tensor(LEFT_VELOCITIES, dtype=torch.int,device = device)
    UP_VELOCITIES = torch.tensor(UP_VELOCITIES, dtype=torch.int,device = device)
    DOWN_VELOCITIES = torch.tensor(DOWN_VELOCITIES, dtype=torch.int,device = device)
    PURE_HORIZONTAL_VELOCITIES = torch.tensor(PURE_HORIZONTAL_VELOCITIES, dtype=torch.int,device = device)
    PURE_VERTICAL_VELOCITIES = torch.tensor(PURE_VERTICAL_VELOCITIES, dtype=torch.int,device = device)
    OPPOSITE = torch.tensor(OPPOSITE, dtype=torch.int,device = device)

def get_density(discrete_velocities):
    return torch.sum(discrete_velocities, dim=-1)

def get_macroscopic_velocity(discrete_velocities, rho):
    return torch.einsum('ijQ,DQ->ijD', discrete_velocities,LATTICE_VELOCITIES)/rho.unsqueeze(-1)

# rho u denotes the macroscopic density and velocity
def get_equilibrium_discrete_velocities(rho, u):
    # project to discrete velocities
    projected_discrete_velocities = torch.einsum("DQ,ijD->ijQ", LATTICE_VELOCITIES, u)
    # normalize the projected discrete velocities
    macroscopic_velocity_magnitude = torch.norm(u, dim=-1,p=2) # jx**2 + jy**2
    # equilibrium distribution
    equilibrium_discrete_velocities = (
    rho.unsqueeze(-1)  # Add a new axis for broadcasting
    * 
    LATTICE_WEIGHTS.unsqueeze(0).unsqueeze(0)  # Add two new axes for broadcasting
    * 
    (
        1
        +
        3 * projected_discrete_velocities
        +
        9/2 * projected_discrete_velocities**2
        -
        3/2 * macroscopic_velocity_magnitude.unsqueeze(-1)**2  # Add a new axis for broadcasting
    )
)
    return equilibrium_discrete_velocities



if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Using Device {device}")
    viscosity = MAX_HORIZONTAL_INFLOW_VELOCITY * 2 * CYLINDER_RADIUS_INDICES / REYNOLDS_NUMBER
    relaxation_omega = 1 / (3 * viscosity + 0.5)
    
    print(f"Relaxation omega: {relaxation_omega}")
    print(f"Viscosity: {viscosity}")
    x,y = torch.arange(N_POINTS_X,device=device), torch.arange(N_POINTS_Y,device=device)
    X,Y = torch.meshgrid(x,y, indexing='ij')
    obstacle_mask = (X - CYLINDER_CENTER_INDEX_X)**2 + (Y - CYLINDER_CENTER_INDEX_Y)**2 < CYLINDER_RADIUS_INDICES**2

    velocity_IC= torch.zeros((N_POINTS_X, N_POINTS_Y, 2),device = device)
    
    velocity_IC[:,:,0] = MAX_HORIZONTAL_INFLOW_VELOCITY

    iter= 0
    def update(discrete_velocities_prev):
        # outflow BC at the right boundary
        discrete_velocities_prev[-1,:,LEFT_VELOCITIES] = discrete_velocities_prev[-2,:,LEFT_VELOCITIES]
        # (2) Macroscopic Velocities
        rho_prev = get_density(discrete_velocities_prev)
        # print("rho in update", rho_prev)
        u_prev = get_macroscopic_velocity(discrete_velocities_prev, rho_prev)
        # print("u in update", u_prev)
        # Inflow Direichlet BC
        u_prev[0,1:-1,:] = velocity_IC[0,1:-1,:]

        # # debug shape
        # print(f"rho_prev {rho_prev.shape}")
        # print("1",get_density(discrete_velocities_prev[0, :, PURE_VERTICAL_VELOCITIES]).shape)
        # print("2",get_density(discrete_velocities_prev[0, :, LEFT_VELOCITIES]).shape)
        # print("u",u_prev[0, :, 0].shape)

        rho_prev[0,:] = get_density(discrete_velocities_prev[0, :, PURE_VERTICAL_VELOCITIES]) + 2 * get_density(discrete_velocities_prev[0, :, LEFT_VELOCITIES]) / (1 - u_prev[0, :, 0])

        
        equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(rho_prev, u_prev)

        discrete_velocities_prev[0,:,RIGHT_VELOCITIES] = equilibrium_discrete_velocities[0,:,RIGHT_VELOCITIES]

        discrete_velocities_post_collision = discrete_velocities_prev-relaxation_omega*(discrete_velocities_prev -equilibrium_discrete_velocities)
        # print(f"shape of discreate velocities post collision{discrete_velocities_post_collision.shape}")
        # print(obstacle_mask)
        masked_rows, masked_cols = torch.where(obstacle_mask)
        for i in range(1, N_DISCRETE_VELOCITIES):
            for row, col in zip(masked_rows, masked_cols):
                discrete_velocities_post_collision[row, col, LATTICE_INDICES[i]] = discrete_velocities_prev[row, col, OPPOSITE[i]]
            # discrete_velocities_post_collision[obstacle_mask,LATTICE_INDICES[i]] = discrete_velocities_prev[obstacle_mask,OPPOSITE[i]]
        discreate_velocities_streamed = discrete_velocities_post_collision.clone()
        for i in range(1, N_DISCRETE_VELOCITIES):
            discreate_velocities_streamed[:,:,i] = torch.roll(torch.roll(discrete_velocities_post_collision[:,:,i], int(LATTICE_VELOCITIES[0,i]), dims=0),int(LATTICE_VELOCITIES[1,i]), dims=1)
        
        return discreate_velocities_streamed
    
    discreate_volocities_prev = get_equilibrium_discrete_velocities(torch.ones((N_POINTS_X, N_POINTS_Y),device = device),velocity_IC)

    for iter in tqdm(range(N_ITERATIONS)):
        discreate_volocities_next = update(discreate_volocities_prev)

        discreate_volocities_prev = discreate_volocities_next
        if iter % PLOT_EVERY_N_STEPS == 0 and VISUALIZE and iter > SKIP_FIRST_N_ITERATIONS:
            density = get_density(discreate_volocities_next)
            macroscopic_velocities = get_macroscopic_velocity(
                discreate_volocities_next,
                density,
            )
            velocity_magnitude = torch.linalg.norm(
                macroscopic_velocities,
                dim=-1,
                ord=2,
            )
            d_u__d_x, d_u__d_y = torch.gradient(macroscopic_velocities[..., 0])
            d_v__d_x, d_v__d_y = torch.gradient(macroscopic_velocities[..., 1])
            curl = (d_u__d_y - d_v__d_x)

            # Velocity Magnitude Contour Plot in the top
            fig,ax= plt.subplots(2,1)
            img1 = ax[0].contourf(
                X,
                Y,
                velocity_magnitude,
                levels=50,
                vmin=-0.02,
                vmax= 0.02,
            )
            plt.colorbar(img1,ax=ax[0])
            ax[0].add_patch(plt.Circle(
                (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDICES,
                color="darkgreen",
            ))

            # Vorticity Magnitude Contour PLot in the bottom
            img2 = ax[1].contourf(
                X,
                Y, 
                curl,
                levels=50,
                cmap = "coolwarm",
                vmin=-0.02,
                vmax= 0.02,
            )
            plt.colorbar(img2,ax=ax[1])
            ax[1].add_patch(plt.Circle(
                (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDICES,
                color="darkgreen",
            ))
            fig.savefig(f"VonKarmon_D2Q9_{iter}.png")