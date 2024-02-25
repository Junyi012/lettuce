import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
lattice = lt.Lattice(lt.D2Q9, device = "cuda", use_native=False)
flow = lt.DecayingTurbulence(resolution=256, reynolds_number=1000, mach_number=0.01, lattice=lattice)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
print("tau:", flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
energyspectrum = lt.EnergySpectrum(lattice, flow)
reporter = lt.ObservableReporter(energyspectrum, interval=500, out=None)
simulation.reporters.append(reporter)

u = lattice.convert_to_numpy(flow.units.convert_velocity_to_pu(lattice.u(simulation.f)))
u_norm = np.linalg.norm(u,axis=0)
plt.figure()
plt.imshow(u_norm)
plt.title('Initialized velocity')
plt.savefig('initialized_velocity_D2Q9.png')

spectrum = flow.energy_spectrum
plt.figure()
plt.loglog(spectrum[1],spectrum[0])
plt.title('Energy spectrum')
plt.xlabel('Wavenumber')
plt.ylabel('Energy')
plt.savefig('energy_spectrum_initialized_D2Q9.png')

simulation.initialize_pressure()
simulation.initialize_f_neq()
mlups = simulation.step(num_steps=15000)
print("Performance in MLUPS:", mlups)

u = lattice.convert_to_numpy(flow.units.convert_velocity_to_pu(lattice.u(simulation.f)))
u_norm = np.linalg.norm(u,axis=0)
plt.figure()
plt.imshow(u_norm)
plt.title('Velocity after simulation')
plt.savefig('velocity_after_simulation_D2Q9.png')

dx = flow.units.convert_length_to_pu(1.0)
grad_u0 = np.gradient(u[0], dx)
grad_u1 = np.gradient(u[1], dx)
vorticity = (grad_u1[0] - grad_u0[1])
plt.figure()
plt.imshow(vorticity, cmap='Spectral')
plt.title('Vorticity after simulation')
plt.savefig('vorticity_after_simulation_D2Q9.png')

spectrum_final = simulation.reporters[0].out[-1]

plt.figure()
plt.loglog(spectrum[1],spectrum[0],label='Initialized spectrum')
plt.loglog(spectrum[1],spectrum_final[2:],label='After simulation')
plt.title('Energy spectrum')
plt.xlabel('Wavenumber')
plt.ylabel('Energy')
plt.ylim(top=1e-1, bottom=1e-8)
plt.legend()
plt.savefig('energy_spectrum_D2Q9.png')