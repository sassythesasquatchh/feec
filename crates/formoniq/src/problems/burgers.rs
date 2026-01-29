use crate::{
  assemble,
  operators::{self, DofIdx},
  problems::nonlinear::{newton_solve, NewtonOptions},
};

use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Vector};
use ddf::cochain::Cochain;
use ddf::whitney::lsf::WhitneyLsf;
use exterior::field::ExteriorField;
use manifold::{
  geometry::coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords},
  geometry::metric::mesh::MeshLengths,
  topology::complex::Complex,
};

pub enum TimeIntegrator {
  ImplicitEuler,
  CrankNicolson,
}

pub struct BurgersResult {
  pub solution: Vec<Cochain>,
}

#[allow(clippy::too_many_arguments)]
pub fn solve_burgers_1d<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  times: &[f64],
  initial_data: Cochain,
  diffusion_coeff: f64,
  boundary_data: Option<F>,
  periodic_pairs: &[(DofIdx, DofIdx)],
  integrator: TimeIntegrator,
  opts: NewtonOptions,
  conservative_form: bool,
) -> BurgersResult
where
  F: Fn(DofIdx) -> f64,
{
  let dim = topology.dim();
  assert_eq!(dim, 1, "Burgers solver currently supports 1D meshes.");

  let mut laplace = assemble::assemble_galmat(
    topology,
    geometry,
    operators::LaplaceBeltramiElmat::new(dim),
  );
  let mut mass = assemble::assemble_galmat(topology, geometry, operators::ScalarMassElmat::new());

  if let Some(ref boundary) = boundary_data {
    let mut dummy = Vector::zeros(mass.nrows());
    assemble::enforce_dirichlet_bc(topology, boundary, &mut laplace, &mut dummy);
    assemble::enforce_dirichlet_bc(topology, boundary, &mut mass, &mut dummy);
  }

  let mut dummy = Vector::zeros(laplace.nrows());
  assemble::enforce_periodic_bc_pairs(&mut laplace, &mut dummy, periodic_pairs);
  let mut dummy = Vector::zeros(mass.nrows());
  assemble::enforce_periodic_bc_pairs(&mut mass, &mut dummy, periodic_pairs);

  let laplace = CsrMatrix::from(&laplace);
  let mass = CsrMatrix::from(&mass);
  let qr = SimplexQuadRule::order3(dim);

  let mut solution = Vec::with_capacity(times.len());
  solution.push(initial_data);

  for window in times.windows(2) {
    let [t0, t1] = [window[0], window[1]];
    let dt = t1 - t0;
    let prev = solution.last().unwrap().coeffs.clone();

    let diffusion = diffusion_coeff;
    let (next, _) = newton_solve(
      prev.clone(),
      |u| {
        let (advection, advection_jac) =
          assemble_burgers_advection(u, topology, geometry, coords, &qr, conservative_form);

        let (residual, jacobian) = match integrator {
          TimeIntegrator::ImplicitEuler => {
            let mut residual = &mass * (u - &prev);
            residual += dt * diffusion * (&laplace * u);
            residual += dt * advection;

            let mut jacobian = CooMatrix::from(&mass);
            add_scaled_coo(&mut jacobian, &CooMatrix::from(&laplace), dt * diffusion);
            add_scaled_coo(&mut jacobian, &advection_jac, dt);
            (residual, jacobian)
          }
          TimeIntegrator::CrankNicolson => {
            let mut residual = &mass * (u - &prev);
            residual += 0.5 * dt * diffusion * (&laplace * (u + &prev));

            let (advection_prev, _) =
              assemble_burgers_advection(&prev, topology, geometry, coords, &qr, conservative_form);
            residual += 0.5 * dt * (advection + advection_prev);

            let mut jacobian = CooMatrix::from(&mass);
            add_scaled_coo(
              &mut jacobian,
              &CooMatrix::from(&laplace),
              0.5 * dt * diffusion,
            );
            add_scaled_coo(&mut jacobian, &advection_jac, 0.5 * dt);
            (residual, jacobian)
          }
        };

        let mut residual = residual;
        let mut jacobian = jacobian;

        if let Some(ref boundary) = boundary_data {
          assemble::enforce_dirichlet_bc(topology, boundary, &mut jacobian, &mut residual);
        }
        assemble::enforce_periodic_bc_pairs(&mut jacobian, &mut residual, periodic_pairs);

        (residual, jacobian)
      },
      opts,
    );

    solution.push(Cochain::new(0, next));
  }

  BurgersResult { solution }
}

fn assemble_burgers_advection(
  coeffs: &Vector,
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: &SimplexQuadRule,
  conservative_form: bool,
) -> (Vector, CooMatrix) {
  let ndofs = topology.skeleton(0).len();
  let mut values = vec![0.0; ndofs];
  let mut rows = Vec::new();
  let mut cols = Vec::new();
  let mut vals = Vec::new();

  for cell in topology.cells().handle_iter() {
    let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
    let geo = geometry.simplex_lengths(cell);
    let dofs: Vec<_> = cell.mesh_subsimps(0).collect();
    let lsfs: Vec<_> = dofs
      .iter()
      .map(|simp| WhitneyLsf::from_coords(cell_coords.clone(), simp.relative_to(&cell)))
      .collect();

    let n = lsfs.len();
    let mut local_res = vec![0.0; n];
    let mut local_jac = vec![0.0; n * n];

    for i in 0..n {
      let res = qr.integrate_local(
        &|local| {
          let global = cell_coords.local2global(local);
          let mut u_val = 0.0;
          let mut u_x = 0.0;
          let mut phi_i = 0.0;
          let mut dphi_i = 0.0;
          for (idof, lsf) in lsfs.iter().enumerate() {
            let phi = lsf.at_point(global.as_view()).coeffs()[0];
            let dphi = lsf.dif().coeffs()[0];
            u_val += coeffs[dofs[idof].kidx()] * phi;
            u_x += coeffs[dofs[idof].kidx()] * dphi;
            if idof == i {
              phi_i = phi;
              dphi_i = dphi;
            }
          }

          if conservative_form {
            -0.5 * u_val * u_val * dphi_i
          } else {
            u_val * u_x * phi_i
          }
        },
        geo.vol(),
      );
      local_res[i] = res;
    }

    for i in 0..n {
      for j in 0..n {
        let jac = qr.integrate_local(
          &|local| {
            let global = cell_coords.local2global(local);
            let mut u_val = 0.0;
            let mut u_x = 0.0;
            let mut phi_i = 0.0;
            let mut dphi_i = 0.0;
            let mut phi_j = 0.0;
            let mut dphi_j = 0.0;
            for (idof, lsf) in lsfs.iter().enumerate() {
              let phi = lsf.at_point(global.as_view()).coeffs()[0];
              let dphi = lsf.dif().coeffs()[0];
              u_val += coeffs[dofs[idof].kidx()] * phi;
              u_x += coeffs[dofs[idof].kidx()] * dphi;
              if idof == i {
                phi_i = phi;
                dphi_i = dphi;
              }
              if idof == j {
                phi_j = phi;
                dphi_j = dphi;
              }
            }

            if conservative_form {
              -u_val * phi_j * dphi_i
            } else {
              u_x * phi_j * phi_i + u_val * dphi_j * phi_i
            }
          },
          geo.vol(),
        );
        local_jac[i * n + j] = jac;
      }
    }

    for i in 0..n {
      let row = dofs[i].kidx();
      values[row] += local_res[i];
      for j in 0..n {
        let col = dofs[j].kidx();
        rows.push(row);
        cols.push(col);
        vals.push(local_jac[i * n + j]);
      }
    }
  }

  let vec = Vector::from_iterator(ndofs, values);
  let jac = CooMatrix::try_from_triplets(ndofs, ndofs, rows, cols, vals).unwrap();
  (vec, jac)
}

fn add_scaled_coo(target: &mut CooMatrix, other: &CooMatrix, scale: f64) {
  for (row, col, val) in other.triplet_iter() {
    target.push(row, col, scale * val);
  }
}
