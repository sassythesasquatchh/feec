use crate::{
  assemble::assemble_galvec, fe::fe_l2_error, io::write_1form_vector_field_vtk,
  operators::SourceElVec, problems::hodge_laplace,
};

use common::{linalg::nalgebra::Vector, util::algebraic_convergence_rate};
use ddf::cochain::cochain_projection;
use exterior::{field::EmbeddedDiffFormClosure, ExteriorElement};
use manifold::geometry::coord::CoordRef;

use std::{
  fs, io,
  path::{Path, PathBuf},
};

const MINOR_RADIUS: f64 = 0.3;
const MAJOR_RADIUS: f64 = 1.0;
const GRADE: usize = 1;
const HOMOLOGY_DIM: usize = 2;
const MAX_RESOLUTION: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct TorusConvergenceRecord {
  pub resolution: usize,
  pub l2_error: f64,
  pub l2_rate: f64,
  pub hd_error: f64,
  pub hd_rate: f64,
}

fn torus_angles(p: CoordRef, major_radius: f64) -> (f64, f64, f64) {
  let x = p[0];
  let y = p[1];
  let z = p[2];

  let s = (x * x + y * y).sqrt();

  let phi = y.atan2(x);
  let theta = z.atan2(s - major_radius);
  let rho = s;

  (theta, phi, rho)
}

fn torus_covectors(p: CoordRef, major_radius: f64) -> (Vector<f64>, Vector<f64>) {
  let x = p[0];
  let y = p[1];
  let z = p[2];

  let s = (x * x + y * y).sqrt();
  let q = (s - major_radius).powi(2) + z * z;

  let dtheta_x = -z * x / (s * q);
  let dtheta_y = -z * y / (s * q);
  let dtheta_z = (s - major_radius) / q;

  let dphi_x = -y / (s * s);
  let dphi_y = x / (s * s);
  let dphi_z = 0.0;

  let dphi = Vector::from_vec(vec![dphi_x, dphi_y, dphi_z]);
  let dtheta = Vector::from_vec(vec![dtheta_x, dtheta_y, dtheta_z]);
  (dtheta, dphi)
}

fn chart_one_form_to_xyz(p: CoordRef, major_radius: f64, a_theta: f64, a_phi: f64) -> Vector<f64> {
  let (dtheta, dphi) = torus_covectors(p, major_radius);
  a_theta * dtheta + a_phi * dphi
}

fn chart_two_form_to_xyz(p: CoordRef, major_radius: f64, coeff_theta_phi: f64) -> Vector<f64> {
  let (dtheta, dphi) = torus_covectors(p, major_radius);
  coeff_theta_phi
    * ExteriorElement::line(dtheta)
      .wedge(&ExteriorElement::line(dphi))
      .into_coeffs()
}

pub fn build_torus_reference_fields() -> (EmbeddedDiffFormClosure, EmbeddedDiffFormClosure) {
  let u_exact = EmbeddedDiffFormClosure::ambient_one_form(
    move |p: CoordRef| {
      let (theta, phi, rho_val) = torus_angles(p, MAJOR_RADIUS);

      let a_theta = 2.0 * (2.0 * theta).cos() * (3.0 * phi).cos()
        - 2.0 * MINOR_RADIUS * theta.cos() / rho_val * (2.0 * phi).cos();

      let a_phi = -3.0 * (2.0 * theta).sin() * (3.0 * phi).sin()
        - rho_val / MINOR_RADIUS * theta.sin() * (2.0 * phi).sin();

      chart_one_form_to_xyz(p, MAJOR_RADIUS, a_theta, a_phi)
    },
    3,
    2,
  );

  let dif_solution_exact = EmbeddedDiffFormClosure::ambient_k_form(
    move |p: CoordRef| {
      let (theta, phi, rho_val) = torus_angles(p, MAJOR_RADIUS);

      let coeff_theta_phi = (theta.sin().powi(2)
        - (rho_val / MINOR_RADIUS + 4.0 * MINOR_RADIUS / rho_val) * theta.cos())
        * (2.0 * phi).sin();

      chart_two_form_to_xyz(p, MAJOR_RADIUS, coeff_theta_phi)
    },
    3,
    2,
    2,
  );

  (u_exact, dif_solution_exact)
}

fn resolve_example_input_path(relative_path: impl AsRef<Path>) -> io::Result<PathBuf> {
  let relative_path = relative_path.as_ref();
  let cwd_candidate = PathBuf::from(relative_path);
  if cwd_candidate.exists() {
    return Ok(cwd_candidate);
  }

  let repo_root_candidate = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("..")
    .join("..")
    .join("..")
    .join(relative_path);
  if repo_root_candidate.exists() {
    return Ok(repo_root_candidate);
  }

  Err(io::Error::new(
    io::ErrorKind::NotFound,
    format!(
      "could not find input {:?}; tried {:?} and {:?}",
      relative_path, cwd_candidate, repo_root_candidate
    ),
  ))
}

pub fn resolution_output_dir(output_dir: impl AsRef<Path>, resolution: usize) -> PathBuf {
  output_dir.as_ref().join(format!("resolution_{resolution}"))
}

pub fn computed_vector_field_output_path(
  output_dir: impl AsRef<Path>,
  resolution: usize,
) -> PathBuf {
  resolution_output_dir(output_dir, resolution).join("solution_computed_vector_field.vtk")
}

pub fn projected_exact_vector_field_output_path(
  output_dir: impl AsRef<Path>,
  resolution: usize,
) -> PathBuf {
  resolution_output_dir(output_dir, resolution).join("solution_projected_exact_vector_field.vtk")
}

pub fn run_torus_convergence(
  output_dir: impl AsRef<Path>,
) -> Result<Vec<TorusConvergenceRecord>, Box<dyn std::error::Error>> {
  let output_dir = output_dir.as_ref();
  let _ = fs::remove_dir_all(output_dir);
  fs::create_dir_all(output_dir)?;

  let (u_exact, dif_solution_exact) = build_torus_reference_fields();

  let f_exact = EmbeddedDiffFormClosure::ambient_one_form(
    move |p: CoordRef| {
      let (theta, phi, rho_val) = torus_angles(p, MAJOR_RADIUS);

      let a = (4.0 / MINOR_RADIUS.powi(2) + 9.0 / rho_val.powi(2)) * (2.0 * theta).sin()
        + 2.0 * theta.sin() * (2.0 * theta).cos() / (MINOR_RADIUS * rho_val);

      let b = (1.0 / MINOR_RADIUS.powi(2) + 4.0 / rho_val.powi(2)) * theta.cos()
        - theta.sin().powi(2) / (MINOR_RADIUS * rho_val);

      let a_prime =
        2.0 * (4.0 / MINOR_RADIUS.powi(2) + 9.0 / rho_val.powi(2)) * (2.0 * theta).cos()
          + 18.0 * MINOR_RADIUS * theta.sin() * (2.0 * theta).sin() / rho_val.powi(3)
          + 2.0 * (theta.cos() * (2.0 * theta).cos() - 2.0 * theta.sin() * (2.0 * theta).sin())
            / (MINOR_RADIUS * rho_val)
          + 2.0 * theta.sin().powi(2) * (2.0 * theta).cos() / rho_val.powi(2);

      let b_prime = 8.0 * MINOR_RADIUS * theta.sin() * theta.cos() / rho_val.powi(3)
        - (1.0 / MINOR_RADIUS.powi(2) + 4.0 / rho_val.powi(2)) * theta.sin()
        - 2.0 * theta.sin() * theta.cos() / (MINOR_RADIUS * rho_val)
        - theta.sin().powi(3) / rho_val.powi(2);

      let f_theta =
        a_prime * (3.0 * phi).cos() - 2.0 * MINOR_RADIUS / rho_val * b * (2.0 * phi).cos();

      let f_phi =
        -3.0 * a * (3.0 * phi).sin() + rho_val / MINOR_RADIUS * b_prime * (2.0 * phi).sin();

      chart_one_form_to_xyz(p, MAJOR_RADIUS, f_theta, f_phi)
    },
    3,
    2,
  );

  let mut errors_l2 = Vec::new();
  let mut errors_hd = Vec::new();
  let mut records = Vec::with_capacity(MAX_RESOLUTION + 1);

  for resolution in 0..=MAX_RESOLUTION {
    let mesh_path =
      resolve_example_input_path(format!("meshes/torus_shell_resolution_{resolution}.msh"))?;
    let mesh_bytes = fs::read(&mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);

    let source_data = assemble_galvec(
      &topology,
      &metric,
      SourceElVec::new(&f_exact, &coords, None),
    );

    let (_, galsol, _) = hodge_laplace::solve_hodge_laplace_source(
      &topology,
      &metric,
      source_data,
      GRADE,
      HOMOLOGY_DIM,
    );

    let u_projected = cochain_projection(&u_exact, &topology, &coords, None);

    let resolution_dir = resolution_output_dir(output_dir, resolution);
    fs::create_dir_all(&resolution_dir)?;
    write_1form_vector_field_vtk(
      computed_vector_field_output_path(output_dir, resolution),
      &coords,
      &topology,
      &galsol,
      "solution_computed_vector_field",
    )?;
    write_1form_vector_field_vtk(
      projected_exact_vector_field_output_path(output_dir, resolution),
      &coords,
      &topology,
      &u_projected,
      "solution_projected_exact_vector_field",
    )?;

    let conv_rate = |errors: &[f64], curr: f64| {
      errors
        .last()
        .map(|&prev| algebraic_convergence_rate(curr, prev))
        .unwrap_or(f64::INFINITY)
    };

    let l2_error = fe_l2_error(&galsol, &u_exact, &topology, &coords);
    let l2_rate = conv_rate(&errors_l2, l2_error);
    errors_l2.push(l2_error);

    let dif_galsol = galsol.dif(&topology);
    let hd_error = fe_l2_error(&dif_galsol, &dif_solution_exact, &topology, &coords);
    let hd_rate = conv_rate(&errors_hd, hd_error);
    errors_hd.push(hd_error);

    records.push(TorusConvergenceRecord {
      resolution,
      l2_error,
      l2_rate,
      hd_error,
      hd_rate,
    });
  }

  Ok(records)
}
