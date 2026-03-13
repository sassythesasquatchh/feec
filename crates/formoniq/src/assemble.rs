use crate::operators::{CoordAwareElMatProvider, DofIdx, ElMatProvider, ElVecProvider};

use common::{
  linalg::nalgebra::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector},
  util,
};
use ddf::CoordSimplexExt;
use exterior::{
  field::{DiffFormClosure, ExteriorField},
  ExteriorGrade,
};
use itertools::Itertools;
use manifold::{
  geometry::{
    coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords, CoordRef},
    metric::{mesh::MeshLengths, simplex::SimplexLengths},
    refsimp_vol,
  },
  topology::{
    complex::Complex,
    handle::{KSimplexIdx, SimplexHandle, SimplexIdx},
    simplex::Simplex,
  },
  Dim,
};

use itertools::izip;
use rayon::prelude::*;
use std::collections::HashSet;

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
  P: Fn(KSimplexIdx) -> bool + Sync,
{
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let entries: Vec<(usize, f64)> = topology
    .boundary_facets()
    .into_par_iter()
    .filter(|fidx| boundary_selector(fidx.kidx))
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
  dim: Dim,
  predicate: P,
) -> Vec<KSimplexIdx>
where
  P: Fn(CoordRef) -> bool + Sync,
{
  assert!(
    dim <= topology.dim() - 1,
    "Simplex dimension exceeds boundary dimension."
  );
  topology
    .boundary_subcomplex_simplices(dim)
    .into_par_iter()
    .filter_map(|simp_idx| {
      let simp = simp_idx.handle(topology);
      let simplex_coords = SimplexCoords::from_simplex_and_coords(&simp, coords);
      let barycenter = simplex_coords.barycenter();
      predicate(barycenter.as_view()).then_some(simp_idx.kidx)
    })
    .collect()
}
// Assemble a boundary (Neumann) Galerkin vector
// $\int_{\Gamma_N} \mathrm{tr}\, \omega \wedge g_N$ on a selectable subset of boundary facets.
pub fn assemble_boundary_integral_term(
  topology: &Complex,
  coords: &MeshCoords,
  test_grade: ExteriorGrade,
  boundary_data: &DiffFormClosure,
  qr: Option<SimplexQuadRule>,
  boundary_selector: &dyn Fn(KSimplexIdx) -> bool,
) -> GalVec {
  let nsimps = topology.skeleton(test_grade).len();
  if nsimps == 0 {
    return Vector::zeros(0);
  }

  let boundary_dim = topology.dim().saturating_sub(1);
  assert!(
    test_grade <= boundary_dim,
    "Test form grade exceeds boundary dimension."
  );
  assert!(
    boundary_data.grade() + test_grade == boundary_dim,
    "Boundary data grade does not match (n-1 - grade(test)). Data grade: {}, test grade: {}, boundary dimension: {}",
    boundary_data.grade(),
    test_grade,
    boundary_dim
  );

  let qr = qr.unwrap_or_else(|| SimplexQuadRule::barycentric(boundary_dim));

  // TODO make safe for parallel execution
  let entries: Vec<(usize, f64)> = topology
    .boundary_facets()
    .into_iter()
    .filter(|fidx| boundary_selector(fidx.kidx))
    .flat_map(|fidx| {
      let facet = fidx.handle(topology);
      let facet_coords = SimplexCoords::from_simplex_and_coords(&facet, coords);

      let orientation_sign = if boundary_dim == 0 {
        1.0
      } else {
        boundary_orientation_sign(facet, coords)
      };

      let elvec = boundary_elvec_for_facet(
        test_grade,
        facet,
        &facet_coords,
        boundary_data,
        &qr,
        orientation_sign,
      );

      facet
        .mesh_subsimps(test_grade)
        .enumerate()
        .map(|(iloc, sub)| (sub.kidx(), elvec[iloc]))
        .collect::<Vec<_>>()
    })
    .collect();

  let mut galvec = Vector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }
  galvec
}

fn boundary_elvec_for_facet(
  test_grade: ExteriorGrade,
  facet: SimplexHandle,
  facet_coords: &SimplexCoords,
  boundary_data: &DiffFormClosure,
  qr: &SimplexQuadRule,
  orientation_sign: f64,
) -> Vector {
  let subs: Vec<_> = facet.mesh_subsimps(test_grade).collect();
  if subs.is_empty() {
    return Vector::zeros(0);
  }

  let mut elvec = Vector::zeros(subs.len());
  let multivector = facet_coords.spanning_multivector();
  let vol = refsimp_vol(facet_coords.dim_intrinsic());

  for (iloc, sub) in subs.iter().enumerate() {
    let local_sub = sub.relative_to(&*facet);
    let lsf = ddf::whitney::lsf::WhitneyLsf::from_coords(facet_coords.clone(), local_sub);
    let f = |xi: CoordRef| {
      let global = facet_coords.local2global(xi);
      let phi = lsf.at_point(global.as_view());
      let g = boundary_data.at_point(global.as_view());
      let integrand = phi.wedge(&g);
      orientation_sign * integrand.apply_form_to_vector(&multivector)
    };
    elvec[iloc] = qr.integrate_local(&f, vol);
  }

  elvec
}

/// Orientation factor for an oriented boundary facet induced by its unique parent cell.
///
/// The factor combines
/// - the sign of the facet in the boundary chain of its parent cell, and
/// - the orientation of that cell with respect to the ambient coordinates.
fn boundary_orientation_sign(facet: SimplexHandle, coords: &MeshCoords) -> f64 {
  let parent_cell = facet
    .cocells()
    .next()
    .expect("Boundary facet should have exactly one parent cell.");

  let facet_sign = parent_cell
    .boundary_chain()
    .find_map(|(sign, subfacet)| (subfacet == facet).then_some(sign))
    .expect("Boundary facet must appear in boundary of its parent cell.")
    .as_f64();

  let parent_coords = SimplexCoords::from_simplex_and_coords(&parent_cell, coords);
  let cell_orientation = parent_coords.orientation().as_f64();

  facet_sign * cell_orientation
}

pub fn drop_boundary_dofs_galmat(complex: &Complex, galmat: &mut GalMat) {
  drop_dofs_galmat(&complex.boundary_vertices().into_iter().collect(), galmat)
}

// Build old-index -> new-index map (None if dropped).
fn build_index_map(n_old: usize, drop: &HashSet<usize>) -> Vec<Option<usize>> {
  let mut map = vec![None; n_old];
  let mut next = 0usize;
  for i in 0..n_old {
    if !drop.contains(&i) {
      map[i] = Some(next);
      next += 1;
    }
  }
  map
}

pub fn drop_dofs_rectangular_galmat(
  drop_rows: &HashSet<usize>,
  drop_cols: &HashSet<usize>,
  galmat: &mut GalMat,
) {
  let nrows_old = galmat.nrows();
  let ncols_old = galmat.ncols();

  assert!(drop_rows.len() <= nrows_old);
  assert!(drop_cols.len() <= ncols_old);
  assert!(drop_rows.iter().all(|&r| r < nrows_old));
  assert!(drop_cols.iter().all(|&c| c < ncols_old));

  let nrows_new = nrows_old - drop_rows.len();
  let ncols_new = ncols_old - drop_cols.len();

  let row_map = build_index_map(nrows_old, drop_rows);
  let col_map = build_index_map(ncols_old, drop_cols);

  let (rows, cols, values) = std::mem::replace(galmat, GalMat::new(0, 0)).disassemble();
  let nnz_old = values.len();

  let mut new_rows = Vec::with_capacity(nnz_old);
  let mut new_cols = Vec::with_capacity(nnz_old);
  let mut new_vals = Vec::with_capacity(nnz_old);

  for (r, c, v) in izip!(rows, cols, values) {
    if let (Some(r2), Some(c2)) = (row_map[r], col_map[c]) {
      new_rows.push(r2);
      new_cols.push(c2);
      new_vals.push(v);
    }
  }

  *galmat = GalMat::try_from_triplets(nrows_new, ncols_new, new_rows, new_cols, new_vals).unwrap();
}

// pub fn drop_dofs_galmat(dofs: &HashSet<DofIdx>, galmat: &mut GalMat) {
//   assert!(galmat.nrows() == galmat.ncols());
//   let ndofs_old = galmat.ncols();
//   let ndofs_new = ndofs_old - dofs.len();

//   let (rows, cols, values) = std::mem::replace(galmat, GalMat::new(0, 0)).disassemble();

//   let (rows, cols, values): (Vec<_>, Vec<_>, Vec<_>) = multizip((rows, cols, values))
//     .filter(|(r, c, _)| !dofs.contains(r) && !dofs.contains(c))
//     .map(|(mut r, mut c, v)| {
//       let diffr = dofs.iter().filter(|&&idof| idof < r).count();
//       let diffc = dofs.iter().filter(|&&idof| idof < c).count();
//       r -= diffr;
//       c -= diffc;
//       (r, c, v)
//     })
//     .multiunzip();

//   *galmat = GalMat::try_from_triplets(ndofs_new, ndofs_new, rows, cols, values).unwrap();
// }

pub fn drop_dofs_galmat(dofs: &HashSet<usize>, galmat: &mut GalMat) {
  drop_dofs_rectangular_galmat(dofs, dofs, galmat);
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

pub fn reintroduce_non_homogenous_dofs_galsols(dof_coeffs: &[(DofIdx, f64)], galsols: &mut Vector) {
  let mut pairs: Vec<(DofIdx, f64)> = dof_coeffs
    .iter()
    .map(|(dof, coeff)| (*dof, *coeff))
    .collect();

  // Sort by dof index
  pairs.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

  let initial_len = pairs.len();
  pairs.dedup_by(|(a, _), (b, _)| a == b);
  let deduped_len = pairs.len();
  assert!(
    deduped_len == initial_len,
    "Duplicate dof indices found in reintroduction of non-homogeneous dofs."
  );

  // Insert rows with the provided coefficient
  let mut owned = std::mem::take(galsols);
  for (dof, coeff) in pairs {
    owned = owned.insert_row(dof, coeff);
  }
  *galsols = owned;
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

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;
  use manifold::geometry::coord::mesh::standard_coord_complex;

  #[test]
  fn boundary_term_interval_endpoints() {
    let (topology, coords) = standard_coord_complex(1);

    let a = 2.0;
    let b = -3.0;

    let boundary_data = DiffFormClosure::scalar(
      move |x| {
        if (x[0]).abs() < 1e-12 {
          a
        } else {
          b
        }
      },
      coords.dim(),
    );

    let v = assemble_boundary_integral_term(&topology, &coords, 0, &boundary_data, None, &|_| true);

    assert_eq!(v.len(), 2);
    assert_abs_diff_eq!(v[0], a, epsilon = 1e-12);
    assert_abs_diff_eq!(v[1], b, epsilon = 1e-12);
  }

  #[test]
  fn boundary_term_single_edge_whitney_integral_is_one() {
    let (topology, coords) = standard_coord_complex(2);

    let g = DiffFormClosure::scalar(|_| 1.0, coords.dim());

    let target_edge_kidx = topology.boundary_facets()[0].kidx;

    let v = assemble_boundary_integral_term(&topology, &coords, 1, &g, None, &|facet_kidx| {
      facet_kidx == target_edge_kidx
    });

    let val = v[target_edge_kidx];

    assert_abs_diff_eq!(val.abs(), 1.0, epsilon = 1e-10);
    for (idx, entry) in v.iter().enumerate() {
      if idx != target_edge_kidx {
        assert_abs_diff_eq!(*entry, 0.0, epsilon = 1e-12);
      }
    }
  }

  #[test]
  fn boundary_term_vertex_hat_integral_edge_length_half() {
    let (topology, coords) = standard_coord_complex(2);

    let target_edge_kidx = topology.boundary_facets()[0].kidx;
    let edge = topology.edges().handle_by_kidx(target_edge_kidx);
    let [v0_idx, v1_idx]: [usize; 2] = (*edge).clone().try_into().unwrap();

    let p0 = coords.coord(v0_idx);
    let p1 = coords.coord(v1_idx);
    let tangent = p1 - p0;
    let edge_length = tangent.norm();
    let unit_tangent = tangent / edge_length;

    let unit_tangent_clone = unit_tangent.clone_owned();
    let boundary_data =
      DiffFormClosure::one_form(move |_| unit_tangent_clone.clone(), coords.dim());

    let v =
      assemble_boundary_integral_term(&topology, &coords, 0, &boundary_data, None, &|facet_kidx| {
        facet_kidx == target_edge_kidx
      });

    assert_eq!(v.len(), topology.skeleton(0).len());
    assert_abs_diff_eq!(v[v0_idx], edge_length / 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(v[v1_idx], edge_length / 2.0, epsilon = 1e-10);

    for (idx, entry) in v.iter().enumerate() {
      if idx != v0_idx && idx != v1_idx {
        assert_abs_diff_eq!(*entry, 0.0, epsilon = 1e-12);
      }
    }
  }
}
