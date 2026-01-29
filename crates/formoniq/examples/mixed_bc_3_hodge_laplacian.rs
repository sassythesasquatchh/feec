use ddf::cochain::{cochain_projection, partial_cochain_projection, Cochain};
use formoniq::{
  assemble::assemble_boundary_integral_term,
  io::{write_1form_vector_field_vtk, write_cochain_vtk},
  operators::InnerProductWeightClosure,
};
use manifold::topology::handle::KSimplexIdx;

use {
  common::{linalg::nalgebra::Vector, util::algebraic_convergence_rate},
  exterior::{field::DiffFormClosure, ExteriorElement},
  formoniq::{
    assemble::assemble_galvec, fe::fe_l2_error, operators::SourceElVec, problems::hodge_laplace,
  },
  manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef},
};

use std::{collections::HashSet, f64::consts::PI, fs, io::Write};

fn write_cochain(path: &str, cochain: &Cochain) -> std::io::Result<()> {
  let mut file = fs::File::create(path)?;
  for coeff in cochain.coeffs.iter() {
    writeln!(file, "{coeff:.12}")?;
  }
  Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();
  let path = "out/examples/general_hodge_laplacian_3";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let solution_exact = DiffFormClosure::one_form(
    |p| {
      let x = p[0];
      let y = p[1];
      let z = p[2];

      // A(x) = x^2 (1-x),  S(t) = sin^2(pi t)
      let a = x * x * (1.0 - x);
      let sy = (PI * y).sin();
      let sz = (PI * z).sin();
      let s2y = sy * sy;
      let s2z = sz * sz;

      let phi = a * s2y * s2z;

      Vector::from_column_slice(&[phi, 0.0, 0.0])
    },
    3,
  );

  let sigma_exact = DiffFormClosure::scalar(
    |p| {
      let x = p[0];
      let y = p[1];
      let z = p[2];

      let ap = 2.0 * x - 3.0 * x * x; // A'(x)
      let sy = (PI * y).sin();
      let sz = (PI * z).sin();
      let s2y = sy * sy;
      let s2z = sz * sz;

      -ap * s2y * s2z
    },
    3,
  );

  // Hodge-Laplacian / forcing: f = d(sigma) + δ(d u)
  let laplacian_exact = DiffFormClosure::one_form(
    |p| {
      let x = p[0];
      let y = p[1];
      let z = p[2];

      let a = x * x * (1.0 - x); // A(x)
      let ap = 2.0 * x - 3.0 * x * x; // A'(x)
      let app = 2.0 - 6.0 * x; // A''(x)

      let sy = (PI * y).sin();
      let sz = (PI * z).sin();
      let s2y = sy * sy;
      let s2z = sz * sz;

      let cos2y = (2.0 * PI * y).cos();
      let cos2z = (2.0 * PI * z).cos();
      let sin2y = (2.0 * PI * y).sin();
      let sin2z = (2.0 * PI * z).sin();

      let fx = -app * s2y * s2z + 2.0 * PI * PI * a * cos2y * s2z + 2.0 * PI * PI * a * s2y * cos2z;

      let fy = -2.0 * PI * ap * sin2y * s2z;
      let fz = -2.0 * PI * ap * s2y * sin2z;

      Vector::from_column_slice(&[fx, fy, fz])
    },
    3,
  );

  let box_mesh = CartesianMeshInfo::new_unit_scaled(3, 4, 1.);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  // ---------------------- Boundary Handling ----------------------
  // let strong_dof_predicate = |p: CoordRef| p[0] == 0.0 || p[1] == 0.0 || p[2] == 0.0;
  // let strong_dof_predicate = |p: CoordRef| p[0] == 0.0;
  let strong_dof_predicate = |p: CoordRef| true;
  let weak_dof_predicate = |p: CoordRef| !strong_dof_predicate(p);

  // let strong_dof_predicate = |_p: CoordRef| true;
  let strong_k_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    1,
    strong_dof_predicate,
  )
  .into_iter()
  .collect::<HashSet<usize>>();

  let strong_k_minus_one_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    0,
    strong_dof_predicate,
  )
  .into_iter()
  .collect::<HashSet<usize>>();

  let strong_k_plus_one_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    2,
    strong_dof_predicate,
  )
  .into_iter()
  .collect::<HashSet<usize>>();

  let strong_k_dof_predicate = |sidx: KSimplexIdx| strong_k_dofs.contains(&sidx);
  let strong_k_minus_one_dof_predicate =
    |sidx: KSimplexIdx| strong_k_minus_one_dofs.contains(&sidx);
  let strong_k_plus_one_dof_predicate = |sidx: KSimplexIdx| strong_k_plus_one_dofs.contains(&sidx);

  let solution_essential_data_map = partial_cochain_projection(
    &solution_exact,
    &topology,
    &coords,
    &strong_k_dof_predicate,
    None,
  );
  let solution_essential_data = |kidx: KSimplexIdx| solution_essential_data_map[&kidx];
  let sigma_essential_data_map = partial_cochain_projection(
    &sigma_exact,
    &topology,
    &coords,
    &strong_k_minus_one_dof_predicate,
    None,
  );
  let sigma_essential_data = |kidx: KSimplexIdx| sigma_essential_data_map[&kidx];

  let weak_face_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    2,
    weak_dof_predicate,
  )
  .into_iter()
  .collect::<HashSet<usize>>();

  let weak_face_dof_predicate = |sidx: KSimplexIdx| weak_face_dofs.contains(&sidx);

  let sigma_neumann_exact = DiffFormClosure::new(
    Box::new(|p| ExteriorElement::new(Vector::from_column_slice(&[0., 0., 0.]), 3, 2)),
    3,
    2,
  );
  let solution_neumann_exact = DiffFormClosure::new(
    Box::new(|p| ExteriorElement::new(Vector::from_column_slice(&[0., 0., 0.]), 3, 1)),
    3,
    1,
  );

  let sigma_neumann_galvec = assemble_boundary_integral_term(
    &topology,
    &coords,
    0,
    &sigma_neumann_exact,
    None,
    &weak_face_dof_predicate,
  );

  let solution_neumann_galvec = assemble_boundary_integral_term(
    &topology,
    &coords,
    1,
    &solution_neumann_exact,
    None,
    &weak_face_dof_predicate,
  );

  // ---------------------------------------------------------------
  let unit_weight = InnerProductWeightClosure::new(|_p| 1.0);

  //   let homology_dim = topology.relative_homology_dim(
  //     1,
  //     &strong_k_minus_one_dof_predicate,
  //     &strong_k_dof_predicate,
  //     &strong_k_plus_one_dof_predicate,
  //   );
  let homology_dim = 0;

  let source_galvec = assemble_galvec(
    &topology,
    &metric,
    SourceElVec::new(&laplacian_exact, &coords, None),
  );

  let (sigma_galsol, u_galsol, _) =
    hodge_laplace::solve_weighted_hodge_laplace_source_with_boundary_conditions(
      &topology,
      &metric,
      None,
      source_galvec,
      1,
      homology_dim,
      &coords,
      None,
      &unit_weight,
      &strong_k_dof_predicate,
      &solution_essential_data,
      &strong_k_minus_one_dof_predicate,
      &sigma_essential_data,
    );

  let u_exact_cochain = cochain_projection(&solution_exact, &topology, &coords, None);
  let sigma_exact_cochain = cochain_projection(&sigma_exact, &topology, &coords, None);
  write_cochain(&format!("{path}/solution_exact.cochain"), &u_exact_cochain)?;
  write_cochain(&format!("{path}/sigma_exact.cochain"), &sigma_exact_cochain)?;
  write_cochain(&format!("{path}/solution_computed.cochain"), &u_galsol)?;
  write_cochain(&format!("{path}/sigma_computed.cochain"), &sigma_galsol)?;

  // VTK outputs for visualization: solution is a 1-form (edge data), sigma is a 0-form (vertex data).
  write_cochain_vtk(
    &format!("{path}/solution_computed.vtk"),
    &coords,
    &topology,
    &u_galsol,
    "solution",
  )?;
  write_cochain_vtk(
    &format!("{path}/solution_exact.vtk"),
    &coords,
    &topology,
    &u_exact_cochain,
    "solution_exact",
  )?;
  write_cochain_vtk(
    &format!("{path}/solution_difference.vtk"),
    &coords,
    &topology,
    &(u_galsol.clone() - u_exact_cochain.clone()),
    "solution_difference",
  )?;
  write_1form_vector_field_vtk(
    &format!("{path}/solution_exact_vector_field.vtk"),
    &coords,
    &topology,
    &u_exact_cochain,
    "solution_exact_vector_field",
  )?;
  write_1form_vector_field_vtk(
    &format!("{path}/solution_computed_vector_field.vtk"),
    &coords,
    &topology,
    &u_galsol,
    "solution_computed_vector_field",
  )?;

  write_cochain_vtk(
    &format!("{path}/sigma_computed.vtk"),
    &coords,
    &topology,
    &sigma_galsol,
    "sigma",
  )?;
  write_cochain_vtk(
    &format!("{path}/sigma_exact.vtk"),
    &coords,
    &topology,
    &sigma_exact_cochain,
    "sigma_exact",
  )?;
  write_cochain_vtk(
    &format!("{path}/sigma_difference.vtk"),
    &coords,
    &topology,
    &(sigma_galsol.clone() - sigma_exact_cochain.clone()),
    "sigma_difference",
  )?;

  let error_l2 = fe_l2_error(&u_galsol, &solution_exact, &topology, &coords);
  println!("Solution L2 error: {error_l2:.6e}");

  let error_l2 = fe_l2_error(&sigma_galsol, &sigma_exact, &topology, &coords);
  println!("Sigma L2 error: {error_l2:.6e}");

  Ok(())
}
