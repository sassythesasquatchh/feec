use std::f64::consts::PI;

use exterior::field::{DiffFormClosure, ExteriorField};
use manifold::gen::cartesian::CartesianMeshInfo;

use ddf::cochain::Cochain;
use formoniq::problems::nonlinear::NewtonOptions;
use formoniq::problems::reaction_diffusion::solve_reaction_diffusion;

fn main() {
  let dim = 2;
  let ncells_axis = 20;
  let mesh_info = CartesianMeshInfo::new_unit(dim, ncells_axis);
  let (topology, coords) = mesh_info.compute_coord_complex();
  let geometry = coords.to_edge_lengths(&topology);

  let terms = 50;
  let exact = DiffFormClosure::scalar(
    move |x| {
      (1..=terms)
        .map(|k| {
          let kf = k as f64;
          (kf.powi(-6)) * (kf * PI * x[0]).sin() * (kf * PI * x[1]).sin()
        })
        .sum::<f64>()
    },
    dim,
  );

  let source = DiffFormClosure::scalar(
    move |x| {
      let series = (1..=terms)
        .map(|k| {
          let kf = k as f64;
          2.0 * PI * PI * kf.powi(-4) * (kf * PI * x[0]).sin() * (kf * PI * x[1]).sin()
        })
        .sum::<f64>();
      let u = (1..=terms)
        .map(|k| {
          let kf = k as f64;
          kf.powi(-6) * (kf * PI * x[0]).sin() * (kf * PI * x[1]).sin()
        })
        .sum::<f64>();
      series + u.powi(3)
    },
    dim,
  );

  let boundary_data = |idx: usize| {
    let x = coords.coord(idx);
    exact.at_point(x).coeffs()[0]
  };

  let initial_guess = Cochain::zero(&topology.skeleton(0));
  let opts = NewtonOptions::standard(10, 1e-5);

  let result = solve_reaction_diffusion(
    &topology,
    &geometry,
    &coords,
    &source,
    boundary_data,
    initial_guess,
    opts,
  );

  println!(
    "Reaction-diffusion solve completed in {} Newton iterations.",
    result.newton_iterations
  );
}
