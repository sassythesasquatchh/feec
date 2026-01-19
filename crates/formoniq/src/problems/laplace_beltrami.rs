//! Module for the Poisson Equation, the prototypical elliptic PDE.

use crate::{
  assemble::{self, GalVec},
  operators::{self, DofCoeff, InnerProductWeightClosure},
};

use common::linalg::{
  faer::FaerCholesky,
  nalgebra::{CsrMatrix, Vector},
  petsc::petsc_ghiep,
};
use ddf::cochain::Cochain;
use manifold::geometry::coord::mesh::MeshCoords;
use manifold::geometry::coord::quadrature::SimplexQuadRule;
use manifold::{
  geometry::metric::mesh::MeshLengths,
  topology::{complex::Complex, handle::KSimplexIdx},
};

pub fn solve_laplace_beltrami_source<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  source_galvec: GalVec,
  essential_boundary_data: F,
  essential_boundary_selector: Option<&dyn Fn(usize) -> bool>,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  solve_laplace_beltrami_source_inner(
    topology,
    geometry,
    source_galvec,
    essential_boundary_data,
    essential_boundary_selector,
    None,
    None,
    None,
  )
}

pub fn solve_laplace_beltrami_source_weighted<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  source_galvec: GalVec,
  essential_boundary_data: F,
  essential_boundary_selector: Option<&dyn Fn(usize) -> bool>,
  mesh_coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  solve_laplace_beltrami_source_inner(
    topology,
    geometry,
    source_galvec,
    essential_boundary_data,
    essential_boundary_selector,
    Some(mesh_coords),
    qr,
    Some(weight),
  )
}

pub fn solve_laplace_beltrami_evp(
  topology: &Complex,
  geometry: &MeshLengths,
  neigen_values: usize,
) -> (Vector, Vec<Cochain>) {
  solve_laplace_beltrami_evp_inner(topology, geometry, neigen_values, None, None, None)
}

pub fn solve_laplace_beltrami_evp_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  neigen_values: usize,
  mesh_coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> (Vector, Vec<Cochain>) {
  solve_laplace_beltrami_evp_inner(
    topology,
    geometry,
    neigen_values,
    Some(mesh_coords),
    qr,
    Some(weight),
  )
}

fn solve_laplace_beltrami_source_inner<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  mut source_galvec: GalVec,
  essential_boundary_data: F,
  essential_boundary_selector: Option<&dyn Fn(usize) -> bool>,
  mesh_coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  let dim = topology.dim();
  let mut laplace_galmat = if let (Some(mesh_coords), Some(weight)) = (mesh_coords, weight) {
    assemble::assemble_galmat_coord_aware(
      topology,
      geometry,
      operators::LaplaceBeltramiElmat::new_weighted(dim, mesh_coords, qr.clone(), weight),
    )
  } else {
    assemble::assemble_galmat(
      topology,
      geometry,
      operators::LaplaceBeltramiElmat::new(dim),
    )
  };
  assemble::enforce_dirichlet_bc_partial(
    topology,
    essential_boundary_data,
    &mut laplace_galmat,
    &mut source_galvec,
    essential_boundary_selector,
  );

  let laplace = CsrMatrix::from(&laplace_galmat);
  let sol = FaerCholesky::new(laplace).solve(&source_galvec);
  Cochain::new(0, sol)
}

/// Eigenvalue problem of Laplace-Beltrami operator.
fn solve_laplace_beltrami_evp_inner(
  topology: &Complex,
  geometry: &MeshLengths,
  neigen_values: usize,
  mesh_coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
) -> (Vector, Vec<Cochain>) {
  let dim = topology.dim();
  let laplace_galmat = if let (Some(mesh_coords), Some(weight)) = (mesh_coords, weight) {
    assemble::assemble_galmat_coord_aware(
      topology,
      geometry,
      operators::LaplaceBeltramiElmat::new_weighted(dim, mesh_coords, qr.clone(), weight),
    )
  } else {
    assemble::assemble_galmat(
      topology,
      geometry,
      operators::LaplaceBeltramiElmat::new(dim),
    )
  };
  let mass_galmat = if let (Some(mesh_coords), Some(weight)) = (mesh_coords, weight) {
    assemble::assemble_galmat_coord_aware(
      topology,
      geometry,
      operators::ScalarMassElmat::new_weighted(mesh_coords, qr.clone(), weight),
    )
  } else {
    assemble::assemble_galmat(topology, geometry, operators::ScalarMassElmat::new())
  };
  let (eigenvals, eigenvecs) = petsc_ghiep(
    &CsrMatrix::from(&laplace_galmat),
    &CsrMatrix::from(&mass_galmat),
    neigen_values,
  );

  let eigenvecs = eigenvecs
    .column_iter()
    .map(|c| Cochain::new(0, c.into_owned()))
    .collect();

  (eigenvals, eigenvecs)
}
