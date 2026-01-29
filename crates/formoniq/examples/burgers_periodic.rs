use std::f64::consts::PI;

use ddf::cochain::Cochain;
use manifold::gen::cartesian::CartesianMeshInfo;

use formoniq::problems::burgers::{solve_burgers_1d, TimeIntegrator};
use formoniq::problems::nonlinear::NewtonOptions;

fn main() {
  let dim = 1;
  let ncells_axis = 50;
  let mesh_info = CartesianMeshInfo::new_unit(dim, ncells_axis);
  let (topology, coords) = mesh_info.compute_coord_complex();
  let geometry = coords.to_edge_lengths(&topology);

  let times: Vec<f64> = (0..=100).map(|i| i as f64 * 0.01).collect();
  let initial = Cochain::from_function(
    |simp| {
      let x = coords.coord(simp.kidx())[0];
      (2.0 * PI * x).sin()
    },
    0,
    &topology,
  );

  let boundary_vertices = topology.boundary_vertices();
  assert_eq!(boundary_vertices.len(), 2, "expected two boundary vertices");
  let mut sorted = boundary_vertices
    .iter()
    .copied()
    .map(|idx| (idx, coords.coord(idx)[0]))
    .collect::<Vec<_>>();
  sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
  let left = sorted[0].0;
  let right = sorted[1].0;

  let periodic_pairs = vec![(right, left)];

  let opts = NewtonOptions::standard(30, 1e-5);

  let result = solve_burgers_1d(
    &topology,
    &geometry,
    &coords,
    &times,
    initial,
    0.01,
    None::<fn(usize) -> f64>,
    &periodic_pairs,
    TimeIntegrator::ImplicitEuler,
    opts,
    true,
  );

  println!("Computed {} time steps.", result.solution.len());
}
