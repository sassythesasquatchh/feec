use crate::operators::{CoordAwareElMatProvider, DofIdx, ElMatProvider, ElVecProvider};

use common::{
  linalg::nalgebra::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector},
  util,
};
use ddf::CoordSimplexExt;
use exterior::{field::DifferentialMultiForm, field::ExteriorField, ExteriorGrade};
use itertools::{multizip, Itertools};
use manifold::{
  geometry::{
    coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords, CoordRef},
    metric::{mesh::MeshLengths, simplex::SimplexLengths},
    refsimp_vol,
  },
  topology::{complex::Complex, handle::SimplexHandle, handle::SimplexIdx, simplex::Simplex},
};

use rayon::prelude::*;
use std::{collections::HashSet, ops::Bound};

pub type GalMat = CooMatrix;

/// Assembly algorithm for the Galerkin Matrix.
fn assemble_galmat_impl<M>(
  topology: &Complex,
  geometry: &MeshLengths,
  row_grade: ExteriorGrade,
  col_grade: ExteriorGrade,
  eval: impl Fn(&SimplexLengths, &Simplex) -> M + Sync,
) -> GalMat
where
  M: std::ops::Index<(usize, usize), Output = f64> + Send,
{
  let nsimps_row = topology.skeleton(row_grade).len();
  let nsimps_col = topology.skeleton(col_grade).len();

  let triplets: Vec<(usize, usize, f64)> = topology
    .cells()
    .handle_iter()
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_lengths(cell);
      let elmat = eval(&geo, &cell);

      let row_subs: Vec<_> = cell.mesh_subsimps(row_grade).collect();
      let col_subs: Vec<_> = cell.mesh_subsimps(col_grade).collect();

      let mut local_triplets = Vec::new();
      for (ilocal, &iglobal) in row_subs.iter().enumerate() {
        for (jlocal, &jglobal) in col_subs.iter().enumerate() {
          let val = elmat[(ilocal, jlocal)];
          if val != 0.0 {
            local_triplets.push((iglobal.kidx(), jglobal.kidx(), val));
          }
        }
      }

      local_triplets
    })
    .collect();

  let (rows, cols, values) = triplets.into_iter().multiunzip();
  GalMat::try_from_triplets(nsimps_row, nsimps_col, rows, cols, values).unwrap()
}

pub fn assemble_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
  elmat: impl ElMatProvider + Sync,
) -> GalMat {
  let (r, c) = (elmat.row_grade(), elmat.col_grade());
  assemble_galmat_impl(topology, geometry, r, c, move |geo, _cell| elmat.eval(geo))
}

pub fn assemble_galmat_coord_aware(
  topology: &Complex,
  geometry: &MeshLengths,
  elmat: impl CoordAwareElMatProvider + Sync,
) -> GalMat {
  let (r, c) = (elmat.row_grade(), elmat.col_grade());
  assemble_galmat_impl(topology, geometry, r, c, move |geo, cell| {
    elmat.eval_with_coords(geo, cell)
  })
}

pub type GalVec = Vector;
/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_galvec(
  topology: &Complex,
  geometry: &MeshLengths,
  elvec: impl ElVecProvider,
) -> GalVec {
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let entries: Vec<(usize, f64)> = topology
    .cells()
    .handle_iter()
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_lengths(cell);
      let elvec = elvec.eval(&geo, &cell);

      let subs: Vec<_> = cell.mesh_subsimps(grade).collect();

      let mut local_entries = Vec::new();
      for (ilocal, &iglobal) in subs.iter().enumerate() {
        if elvec[ilocal] != 0.0 {
          local_entries.push((iglobal.kidx(), elvec[ilocal]));
        }
      }

      local_entries
    })
    .collect();

  let mut galvec = Vector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }
  galvec
}

/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_boundary_galvec<P>(
  topology: &Complex,
  geometry: &MeshLengths,
  elvec: impl ElVecProvider,
  boundary_selector: P,
) -> GalVec
where
  P: Fn(SimplexIdx) -> bool + Sync,
{
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let entries: Vec<(usize, f64)> = topology
    .boundary_facets()
    .into_par_iter()
    .filter(|fidx| boundary_selector(*fidx))
    .flat_map(|fidx| {
      let facet = fidx.handle(topology);
      let geo = geometry.simplex_lengths(facet);
      let elvec = elvec.eval(&geo, &facet);

      let subs: Vec<_> = facet.mesh_subsimps(grade).collect();

      let mut local_entries = Vec::new();
      for (ilocal, &iglobal) in subs.iter().enumerate() {
        if elvec[ilocal] != 0.0 {
          local_entries.push((iglobal.kidx(), elvec[ilocal]));
        }
      }

      local_entries
    })
    .collect();

  let mut galvec = Vector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }
  galvec
}

/// Return simplices of a given grade whose barycenter satisfies a predicate.
///
/// Useful for partitioning boundaries by geometric location.
pub fn boundary_simplices_where_barycenter<P>(
  topology: &Complex,
  coords: &MeshCoords,
  grade: ExteriorGrade,
  predicate: P,
) -> Vec<SimplexIdx>
where
  P: Fn(CoordRef) -> bool + Sync,
{
  assert!(
    grade <= topology.dim() - 1,
    "Grade exceeds boundary dimension."
  );
  topology
    .boundary_subcomplex_simplices(grade)
    .into_par_iter()
    .filter_map(|simp_idx| {
      let simp = simp_idx.handle(topology);
      let simplex_coords = SimplexCoords::from_simplex_and_coords(&simp, coords);
      let barycenter = simplex_coords.barycenter();
      predicate(barycenter.as_view()).then_some(simp_idx)
    })
    .collect()
}

/// Assemble a boundary (Neumann) Galerkin vector
/// $\int_{\Gamma_N} \mathrm{tr}\, \omega \wedge g_N$ on a selectable subset of boundary facets.
// pub fn assemble_boundary_galvec<G, P>(
//   topology: &Complex,
//   coords: &MeshCoords,
//   test_grade: ExteriorGrade,
//   boundary_data: &G,
//   qr: Option<SimplexQuadRule>,
//   boundary_selector: P,
// ) -> GalVec
// where
//   G: DifferentialMultiForm + Sync,
//   P: Fn(SimplexIdx) -> bool + Sync,
// {
//   let nsimps = topology.skeleton(test_grade).len();
//   if nsimps == 0 {
//     return Vector::zeros(0);
//   }

//   let boundary_dim = topology.dim().saturating_sub(1);
//   assert!(
//     test_grade <= boundary_dim,
//     "Test form grade exceeds boundary dimension."
//   );
//   assert!(
//     boundary_data.grade() + test_grade == boundary_dim,
//     "Boundary data grade does not match (n-1 - grade(test)). Data grade: {}, test grade: {}, boundary dimension: {}",
//     boundary_data.grade(),
//     test_grade,
//     boundary_dim
//   );

//   let qr = qr.unwrap_or_else(|| SimplexQuadRule::barycentric(boundary_dim));

//   let entries: Vec<(usize, f64)> = topology
//     .boundary_facets()
//     .into_par_iter()
//     .filter(|fidx| boundary_selector(*fidx))
//     .flat_map(|fidx| {
//       let facet = fidx.handle(topology);
//       let facet_coords = SimplexCoords::from_simplex_and_coords(&facet, coords);
//       let elvec = boundary_elvec_for_facet(test_grade, facet, &facet_coords, boundary_data, &qr);

//       facet
//         .mesh_subsimps(test_grade)
//         .enumerate()
//         .map(|(iloc, sub)| (sub.kidx(), elvec[iloc]))
//         .collect::<Vec<_>>()
//     })
//     .collect();

//   let mut galvec = Vector::zeros(nsimps);
//   for (irow, val) in entries {
//     galvec[irow] += val;
//   }
//   galvec
// }

// fn boundary_elvec_for_facet<G: DifferentialMultiForm>(
//   test_grade: ExteriorGrade,
//   facet: SimplexHandle,
//   facet_coords: &SimplexCoords,
//   boundary_data: &G,
//   qr: &SimplexQuadRule,
// ) -> Vector {
//   let subs: Vec<_> = facet.mesh_subsimps(test_grade).collect();
//   if subs.is_empty() {
//     return Vector::zeros(0);
//   }

//   let mut elvec = Vector::zeros(subs.len());
//   let multivector = facet_coords.spanning_multivector();
//   let vol = refsimp_vol(facet_coords.dim_intrinsic());

//   for (iloc, sub) in subs.iter().enumerate() {
//     // Convert sub-simplex to local vertex numbering for this facet so barycentric lookup is valid.
//     let local_sub = sub.relative_to(&*facet);
//     let lsf = ddf::whitney::lsf::WhitneyLsf::from_coords(facet_coords.clone(), local_sub);
//     let f = |xi: CoordRef| {
//       let global = facet_coords.local2global(xi);
//       let phi = lsf.at_point(global.as_view());
//       let g = boundary_data.at_point(global.as_view());
//       let integrand = phi.wedge(&g);
//       integrand.apply_form_to_vector(&multivector)
//     };
//     elvec[iloc] = qr.integrate_local(&f, vol);
//   }

//   elvec
// }

pub fn drop_boundary_dofs_galmat(complex: &Complex, galmat: &mut GalMat) {
  drop_dofs_galmat(&complex.boundary_vertices().into_iter().collect(), galmat)
}

pub fn drop_dofs_galmat(dofs: &HashSet<DofIdx>, galmat: &mut GalMat) {
  assert!(galmat.nrows() == galmat.ncols());
  let ndofs_old = galmat.ncols();
  let ndofs_new = ndofs_old - dofs.len();

  let (rows, cols, values) = std::mem::replace(galmat, GalMat::new(0, 0)).disassemble();

  let (rows, cols, values): (Vec<_>, Vec<_>, Vec<_>) = multizip((rows, cols, values))
    .filter(|(r, c, _)| !dofs.contains(r) && !dofs.contains(c))
    .map(|(mut r, mut c, v)| {
      let diffr = dofs.iter().filter(|&&idof| idof < r).count();
      let diffc = dofs.iter().filter(|&&idof| idof < c).count();
      r -= diffr;
      c -= diffc;
      (r, c, v)
    })
    .multiunzip();

  *galmat = GalMat::try_from_triplets(ndofs_new, ndofs_new, rows, cols, values).unwrap();
}

pub fn drop_dofs_galvec(dofs: &[DofIdx], galvec: &mut GalVec) {
  *galvec = std::mem::take(galvec).remove_rows_at(dofs);
}

pub fn reintroduce_boundary_dofs_galsols(complex: &Complex, galsols: &mut Matrix) {
  reintroduce_dropped_dofs_galsols(complex.boundary_vertices(), galsols)
}

pub fn reintroduce_dropped_dofs_galsols(mut dofs: Vec<DofIdx>, galsols: &mut Matrix) {
  dofs.sort_unstable();
  dofs.dedup();

  let mut galsol_owned = std::mem::take(galsols);
  for dof in dofs {
    galsol_owned = galsol_owned.insert_row(dof, 0.0);
  }
  *galsols = galsol_owned;
}

pub fn enforce_homogeneous_dirichlet_bc(
  complex: &Complex,
  galmat: &mut GalMat,
  galvec: &mut Vector,
) {
  fix_dofs_zero(&complex.boundary_vertices(), galmat, galvec);
}

pub fn enforce_dirichlet_bc_partial<F>(
  complex: &Complex,
  boundary_coeff_map: F,
  galmat: &mut GalMat,
  galvec: &mut Vector,
  boundary_selector: Option<&dyn Fn(usize) -> bool>,
) where
  F: Fn(DofIdx) -> f64,
{
  let boundary_selector = boundary_selector.unwrap_or(&|_: usize| true);
  let boundary_dofs = complex.boundary_vertices();
  let dof_coeffs: Vec<_> = boundary_dofs
    .into_iter()
    .filter(|sidx| boundary_selector(*sidx))
    .map(|idof| (idof, boundary_coeff_map(idof)))
    .collect();

  fix_dofs_coeff(&dof_coeffs, galmat, galvec);
}

pub fn enforce_dirichlet_bc<F>(
  complex: &Complex,
  boundary_coeff_map: F,
  galmat: &mut GalMat,
  galvec: &mut Vector,
) where
  F: Fn(DofIdx) -> f64,
{
  enforce_dirichlet_bc_partial(complex, boundary_coeff_map, galmat, galvec, None);
}

pub fn enforce_essential_bc<F>(
  grade: ExteriorGrade,
  complex: &Complex,
  boundary_coeff_map: F,
  galmat: &mut GalMat,
  galvec: &mut Vector,
  boundary_selector: Option<&dyn Fn(SimplexIdx) -> bool>,
) where
  F: Fn(DofIdx) -> f64,
{
  let boundary_selector = boundary_selector.unwrap_or(&|_sidx: SimplexIdx| true);
  let boundary_dofs = complex.boundary_subcomplex_simplices(grade);
  let dof_coeffs: Vec<_> = boundary_dofs
    .into_iter()
    .filter(|sidx| boundary_selector(*sidx))
    .map(|simp| {
      let idof = simp.kidx;
      (idof, boundary_coeff_map(idof))
    })
    .collect();
  fix_dofs_coeff(&dof_coeffs, galmat, galvec);
}

pub fn fix_dofs_zero(dofs: &[DofIdx], galmat: &mut GalMat, galvec: &mut Vector) {
  let ndofs = galmat.nrows();
  let dof_flags = util::indicies_to_flags(dofs, ndofs);
  galmat.set_zero(|i, j| dof_flags[i] || dof_flags[j]);
  for &idof in dofs {
    galmat.push(idof, idof, 1.0);
    galvec[idof] = 0.0;
  }
}

/// Fix DOFs of FE solution.
///
/// Modifies supplied galerkin matrix and galerkin vector,
/// such that the FE solution has the optionally given coefficents on the dofs.
/// $mat(A_0, 0; 0, I) vec(mu_0, mu_diff) = vec(phi - A_(0 diff) gamma, gamma)$
pub fn fix_dofs_coeff(dof_coeffs: &[(DofIdx, f64)], galmat: &mut GalMat, galvec: &mut Vector) {
  let ndofs = galmat.nrows();

  let dof_coeffs_opt = util::sparse_to_dense_data(dof_coeffs.to_vec(), ndofs);
  let dof_coeffs_zeroed =
    Vector::from_iterator(ndofs, dof_coeffs_opt.iter().map(|v| v.unwrap_or(0.0)));

  // Modify galvec.
  let galmat_csr = CsrMatrix::from(&*galmat);
  *galvec -= galmat_csr * dof_coeffs_zeroed;

  // Set galvec to prescribed coefficents.
  dof_coeffs.iter().for_each(|&(i, v)| galvec[i] = v);

  // Set entires zero that share a (row or column) index with a fixed dof.
  galmat.set_zero(|r, c| dof_coeffs_opt[r].is_some() || dof_coeffs_opt[c].is_some());

  // Set galmat diagonal for dofs to one.
  for &(i, _) in dof_coeffs {
    galmat.push(i, i, 1.0);
  }
}

/// $mat(A_0, A_(0 diff); 0, I) vec(mu_0, mu_diff) = vec(phi, gamma)$
//#[allow(unused_variables, unreachable_code)]
pub fn fix_dofs_coeff_alt(dof_coeffs: &[(DofIdx, f64)], galmat: &mut GalMat, galvec: &mut Vector) {
  tracing::warn!("use of `fix_dofs_coeff_alt` probably doesn't work.");

  let ndofs = galmat.nrows();
  let dof_coeffs_opt = util::sparse_to_dense_data(dof_coeffs.to_vec(), ndofs);

  // Set entires zero that share a row index with a fixed dof.
  galmat.set_zero(|r, _| dof_coeffs_opt[r].is_some());

  // Set galmat diagonal for dofs to one.
  for &(i, _) in dof_coeffs {
    galmat.push(i, i, 1.0);
  }

  // Set galvec to prescribed coefficents.
  for &(i, v) in dof_coeffs.iter() {
    galvec[i] = v
  }
}
