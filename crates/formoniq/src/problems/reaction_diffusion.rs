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

pub struct ReactionDiffusionResult {
  pub solution: Cochain,
  pub newton_iterations: usize,
}

pub fn solve_reaction_diffusion<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  source: &F,
  boundary_data: impl Fn(DofIdx) -> f64,
  initial_guess: Cochain,
  opts: NewtonOptions,
) -> ReactionDiffusionResult
where
  F: ExteriorField + Sync,
{
  let dim = topology.dim();
  let mut laplace = assemble::assemble_galmat(
    topology,
    geometry,
    operators::LaplaceBeltramiElmat::new(dim),
  );

  let source_vec = assemble::assemble_galvec(
    topology,
    geometry,
    operators::SourceElVec::new(source, coords, Some(SimplexQuadRule::order3(dim))),
  );

  let mut rhs = source_vec;
  assemble::enforce_dirichlet_bc(topology, &boundary_data, &mut laplace, &mut rhs);

  let laplace = CsrMatrix::from(&laplace);
  let qr = SimplexQuadRule::order3(dim);

  let (solution, status) = newton_solve(
    initial_guess.coeffs,
    |u| {
      let (reaction, reaction_jac) = assemble_reaction(u, topology, geometry, coords, &qr);
      let mut residual = &laplace * u + reaction - &rhs;
      let mut jacobian = CooMatrix::from(&laplace);
      add_scaled_coo(&mut jacobian, &reaction_jac, 1.0);
      assemble::enforce_dirichlet_bc(topology, &boundary_data, &mut jacobian, &mut residual);
      (residual, jacobian)
    },
    opts,
  );

  let iterations = match status {
    super::nonlinear::NonlinearSolveStatus::Converged { iterations, .. } => iterations,
    super::nonlinear::NonlinearSolveStatus::MaxIters { iterations, .. } => iterations,
  };

  ReactionDiffusionResult {
    solution: Cochain::new(0, solution),
    newton_iterations: iterations,
  }
}

fn assemble_reaction(
  coeffs: &Vector,
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: &SimplexQuadRule,
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
          let mut phi_i = 0.0;
          for (idof, lsf) in lsfs.iter().enumerate() {
            let phi = lsf.at_point(global.as_view()).coeffs()[0];
            u_val += coeffs[dofs[idof].kidx()] * phi;
            if idof == i {
              phi_i = phi;
            }
          }
          u_val * u_val * u_val * phi_i
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
            let mut phi_i = 0.0;
            let mut phi_j = 0.0;
            for (idof, lsf) in lsfs.iter().enumerate() {
              let phi = lsf.at_point(global.as_view()).coeffs()[0];
              u_val += coeffs[dofs[idof].kidx()] * phi;
              if idof == i {
                phi_i = phi;
              }
              if idof == j {
                phi_j = phi;
              }
            }
            3.0 * u_val * u_val * phi_i * phi_j
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
