use std::f64::consts::PI;

use common::linalg::nalgebra::Vector;
use ddf::cochain::Cochain;
use manifold::gen::cartesian::CartesianMeshInfo;

use formoniq::problems::burgers::{solve_burgers_1d, TimeIntegrator};
use formoniq::problems::nonlinear::NewtonOptions;

fn main() {
  let ncells_axis = 100;
  let mesh_info = CartesianMeshInfo::new_min_max(
    Vector::from_element(1, -1.0),
    Vector::from_element(1, 1.0),
    ncells_axis,
  );
  let (topology, coords) = mesh_info.compute_coord_complex();
  let geometry = coords.to_edge_lengths(&topology);

  let dt = 0.02;
  let times: Vec<f64> = (0..=50).map(|i| i as f64 * dt).collect();
  let initial = Cochain::from_function(
    |simp| {
      let x = coords.coord(simp.kidx())[0];
      -(PI * x).sin()
    },
    0,
    &topology,
  );

  let boundary_data = |idx: usize| {
    let x = coords.coord(idx)[0];
    if x.abs() >= 1.0 - 1e-12 {
      0.0
    } else {
      0.0
    }
  };

  let opts = NewtonOptions::standard(30, 1e-5);

  let result = solve_burgers_1d(
    &topology,
    &geometry,
    &coords,
    &times,
    initial,
    0.001,
    Some(boundary_data),
    &[],
    TimeIntegrator::CrankNicolson,
    opts,
    false,
  );

  println!("Computed {} time steps.", result.solution.len());
}
