// use exterior::field::DiffFormClosure;
// use formoniq::{
//   operators::InnerProductWeightClosure,
//   problems::hodge_laplace::solve_weighted_hodge_laplace_source,
// };
// use manifold::gen::cartesian::CartesianMeshInfo;
// use std::{f64::consts::PI, fs};

// use formoniq::fe::fe_l2_error;

// fn main() {
//   tracing_subscriber::fmt::init();
//   let path = "out/electrodynamics/magnetostatic_poisson";
//   let _ = fs::remove_dir_all(path);
//   fs::create_dir_all(path).unwrap();

//   let permittivity: f64 = 8.854e-12;

//   let grade = 1;
//   let homology_dim = 0;
//   let dim = 3;

//   println!("Solving Electrostatic Poisson in {dim}d.");

//   let solution_exact =
//     DiffFormClosure::scalar(|p| p.iter().map(|&pi| (PI * pi).sin()).product(), dim);

//   let rhs = DiffFormClosure::scalar(
//     move |p| 3. * PI * PI * permittivity * p.iter().map(|&pi| (PI * pi).sin()).product::<f64>(),
//     dim,
//   );

//   let inner_product_weight = InnerProductWeightClosure::new(move |_p| permittivity);
//   let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, 10, 1.);

//   let (topology, coords) = box_mesh.compute_coord_complex();
//   let metric = coords.to_edge_lengths(&topology);

//   let load_vector = formoniq::assemble::assemble_galvec(
//     &topology,
//     &metric,
//     formoniq::operators::SourceElVec::new_weighted(&rhs, &coords, None, &inner_product_weight),
//   );

//   let (_, galsol, _) = solve_weighted_hodge_laplace_source(
//     &topology,
//     &metric,
//     load_vector,
//     grade,
//     homology_dim,
//     &coords,
//     None,
//     &inner_product_weight,
//   );

//   let error_l2 = fe_l2_error(&galsol, &solution_exact, &topology, &coords);
//   println!("L2 error: {error_l2:.6e}");
// }

fn main() {}
