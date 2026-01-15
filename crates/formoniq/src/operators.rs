use {
  common::{
    combo::{factorial, Sign},
    linalg::nalgebra::{Matrix, Vector},
  },
  ddf::{whitney::lsf::WhitneyLsf, ManifoldComplexExt},
  exterior::{
    field::ExteriorField, list::ExteriorElementList, term::multi_gramian, Dim, ExteriorGrade,
  },
  manifold::{
    geometry::{
      coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords, CoordRef},
      metric::simplex::SimplexLengths,
    },
    topology::{
      complex::Complex,
      simplex::{standard_subsimps, Simplex},
    },
  },
  std::{
    ops::{AddAssign, Mul},
    sync::Arc,
  },
};

pub type DofIdx = usize;
pub type DofCoeff = f64;

pub type ElMat = Matrix;
pub trait ElMatProviderBase: Sync {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
}

pub trait ElMatProvider: ElMatProviderBase {
  fn eval(&self, geometry: &SimplexLengths) -> ElMat;
}

pub trait CoordAwareElMatProvider: ElMatProviderBase {
  fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> ElMat;
}

pub struct InnerProductWeightClosure<T = f64>
where
  T: AddAssign + Mul<f64, Output = T>,
{
  f: Arc<dyn Fn(CoordRef) -> T + Sync + Send>,
}

impl<T> InnerProductWeightClosure<T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  pub fn new<F>(f: F) -> Self
  where
    F: Fn(CoordRef) -> T + Sync + Send + 'static,
  {
    Self { f: Arc::new(f) }
  }

  pub fn apply(&self, x: CoordRef, coeffs: &Vector) -> Vector {
    (self.f)(x).apply(coeffs)
  }

  pub fn apply_batched(&self, x: CoordRef, coeffs: &Matrix) -> Matrix {
    (self.f)(x).apply_batched(coeffs)
  }
}

pub trait ApplyWeight: Send + Sync {
  fn apply_batched(&self, coeffs: &Matrix) -> Matrix;
  fn apply(&self, coeffs: &Vector) -> Vector;
}

impl ApplyWeight for f64 {
  fn apply_batched(&self, coeffs: &Matrix) -> Matrix {
    *self * coeffs
  }
  fn apply(&self, coeffs: &Vector) -> Vector {
    *self * coeffs
  }
}

impl ApplyWeight for Matrix {
  fn apply_batched(&self, coeffs: &Matrix) -> Matrix {
    self * coeffs
  }
  fn apply(&self, coeffs: &Vector) -> Vector {
    self * coeffs
  }
}

struct InnerProductWeight<T: ApplyWeight> {
  weight: T,
}

impl<T: ApplyWeight> InnerProductWeight<T> {
  pub fn apply_batched(&self, coeffs: &Matrix) -> Matrix {
    self.weight.apply_batched(coeffs)
  }
}

fn apply_optional_weight<T: ApplyWeight>(
  weight: Option<&InnerProductWeight<T>>,
  coeffs: &Matrix,
) -> Matrix {
  if let Some(w) = weight {
    w.apply_batched(coeffs)
  } else {
    coeffs.clone()
  }
}

fn scalar_cell_weight(
  cell: &Simplex,
  weight_function: Option<&InnerProductWeightClosure<f64>>,
  coords: Option<&MeshCoords>,
  qr: Option<&SimplexQuadRule>,
) -> f64 {
  let weight =
    if let (Some(weight_function), Some(coords), Some(qr)) = (weight_function, coords, qr) {
      let cell_coords = SimplexCoords::from_simplex_and_coords(cell, coords);
      // vol is set to 1 because we are not integrating over the simplex volume here,
      // we simply want the average value of the weight function over the simplex.
      qr.integrate_local(
        &|local: CoordRef| {
          let global = cell_coords.local2global(local);
          (weight_function.f)(global.as_view())
        },
        1.0,
      )
    } else {
      1.0
    };

  weight
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
///
/// $A = [(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct LaplaceBeltramiElmat<'a> {
  dim: Dim,
  ref_difbarys: Matrix,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<f64>>,
}
impl<'a> LaplaceBeltramiElmat<'a> {
  pub fn new(dim: Dim) -> Self {
    let ref_difbarys = SimplexCoords::standard(dim).difbarys().transpose();
    Self {
      dim,
      ref_difbarys,
      coords: None,
      qr: None,
      weight: None,
    }
  }

  pub fn new_weighted(
    dim: Dim,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<f64>,
  ) -> Self {
    let ref_difbarys = SimplexCoords::standard(dim).difbarys().transpose();
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self {
      dim,
      ref_difbarys,
      coords: Some(coords),
      qr: Some(qr),
      weight: Some(weight),
    }
  }
}

impl<'a> ElMatProviderBase for LaplaceBeltramiElmat<'a> {
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
    0
  }
}
impl<'a> ElMatProvider for LaplaceBeltramiElmat<'a> {
  fn eval(&self, geometry: &SimplexLengths) -> ElMat {
    assert!(self.dim == geometry.dim());
    geometry.vol()
      * geometry
        .to_metric_tensor()
        .inverse()
        .norm_sq_mat(&self.ref_difbarys)
  }
}

impl<'a> CoordAwareElMatProvider for LaplaceBeltramiElmat<'a> {
  fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> ElMat {
    scalar_cell_weight(cell, self.weight, self.coords, self.qr.as_ref()) * self.eval(geometry)
  }
}

/// Exact Element Matrix Provider for scalar mass bilinear form.
pub struct ScalarMassElmat<'a> {
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<f64>>,
}
impl<'a> ScalarMassElmat<'a> {
  pub fn new() -> Self {
    Self {
      coords: None,
      qr: None,
      weight: None,
    }
  }
}

impl<'a> ScalarMassElmat<'a> {
  pub fn new_weighted(
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<f64>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(coords.dim()));
    Self {
      coords: Some(coords),
      qr: Some(qr),
      weight: Some(weight),
    }
  }
}
impl<'a> ElMatProviderBase for ScalarMassElmat<'a> {
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
    0
  }
}
impl<'a> ElMatProvider for ScalarMassElmat<'a> {
  fn eval(&self, geometry: &SimplexLengths) -> ElMat {
    let ndofs = geometry.nvertices();
    let dim = geometry.dim();
    let v = geometry.vol() / ((dim + 1) * (dim + 2)) as f64;
    let mut elmat = Matrix::from_element(ndofs, ndofs, v);
    elmat.fill_diagonal(2.0 * v);
    elmat
  }
}

impl<'a> CoordAwareElMatProvider for ScalarMassElmat<'a> {
  fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> ElMat {
    scalar_cell_weight(cell, self.weight, self.coords, self.qr.as_ref()) * self.eval(geometry)
  }
}

/// Approximated Element Matrix Provider for scalar mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub struct ScalarLumpedMassElmat;
impl ElMatProviderBase for ScalarLumpedMassElmat {
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
    0
  }
}
impl ElMatProvider for ScalarLumpedMassElmat {
  fn eval(&self, geomery: &SimplexLengths) -> ElMat {
    let n = geomery.nvertices();
    let v = geomery.vol() / n as f64;
    Matrix::from_diagonal_element(n, n, v)
  }
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form.
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct HodgeMassElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  dim: Dim,
  grade: ExteriorGrade,
  simplices: Vec<Simplex>,
  wedge_terms: Vec<ExteriorElementList>,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<T>>,
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> HodgeMassElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let simplices: Vec<_> = standard_subsimps(dim, grade).collect();
    let wedge_terms: Vec<ExteriorElementList> = simplices
      .iter()
      .cloned()
      .map(|simp| WhitneyLsf::standard(dim, simp).wedge_terms().collect())
      .collect();

    Self {
      dim,
      grade,
      simplices,
      wedge_terms,
      coords,
      qr,
      weight,
    }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let scalar_mass = ScalarMassElmat::new().eval(geometry);
    let mut elmat = Matrix::zeros(self.simplices.len(), self.simplices.len());

    let weight_to_apply = if let Some(weight) = &self.weight {
      let topology = topology
        .expect("Weighted HodgeMassElmat requires a cell (topology) to evaluate the weight.");
      let qr = self
        .qr
        .as_ref()
        .expect("Inner product weight provided, but no quadrature rule specified.");
      let coords = self
        .coords
        .as_ref()
        .expect("Inner product weight provided, but no mesh coordinates specified.");

      let cell_coords = SimplexCoords::from_simplex_and_coords(topology, coords);

      // vol is set to 1 because we are not integrating over the simplex volume here,
      // we simply want the average value of the weight function over the simplex.
      let quadrature_result = qr.integrate_local(
        &|local: CoordRef| {
          let global = cell_coords.local2global(local);
          (weight.f)(global.as_view())
        },
        1.0,
      );

      Some(InnerProductWeight {
        weight: quadrature_result,
      })
    } else {
      None
    };

    for (i, asimp) in self.simplices.iter().enumerate() {
      for (j, bsimp) in self.simplices.iter().enumerate() {
        let wedge_terms_a = &self.wedge_terms[i];
        let wedge_terms_b = &self.wedge_terms[j];
        let wedge_inners = multi_gramian(&geometry.to_metric_tensor().inverse(), self.grade)
          .inner_mat(
            &apply_optional_weight(weight_to_apply.as_ref(), wedge_terms_a.coeffs()),
            wedge_terms_b.coeffs(),
          );

        let nvertices = self.grade + 1;
        let mut sum = 0.0;
        for avertex in 0..nvertices {
          for bvertex in 0..nvertices {
            let sign = Sign::from_parity(avertex + bvertex);

            let inner = wedge_inners[(avertex, bvertex)];

            sum += sign.as_f64() * inner * scalar_mass[(asimp[avertex], bsimp[bvertex])];
          }
        }

        elmat[(i, j)] = sum;
      }
    }

    factorial(self.grade).pow(2) as f64 * elmat
  }
}

impl<'a> HodgeMassElmat<'a, f64> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }
}

impl<'a, T> ElMatProviderBase for HodgeMassElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn row_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.grade
  }
}

impl<'a, T> CoordAwareElMatProvider for HodgeMassElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> ElMat {
    self._eval(geometry, Some(cell))
  }
}

impl<'a, T> ElMatProvider for HodgeMassElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn eval(&self, geometry: &SimplexLengths) -> ElMat {
    // This is only valid for the unweighted case
    debug_assert!(self.weight.is_none());
    self._eval(geometry, None)
  }
}
/// Element Matrix Provider for the weak mixed exterior derivative $(dif sigma, v)$.
///
/// $A = [inner(dif lambda_J, lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_, J in Delta_(k-1) (K))$
pub struct DifElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  mass: HodgeMassElmat<'a, T>,
  dif: Matrix,
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> DifElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  pub fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let mass = HodgeMassElmat::_new(dim, grade, coords, qr, weight);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade - 1);
    let dif = Matrix::from(&dif);
    Self { mass, dif }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    let mass = self.mass._eval(geometry, topology);
    mass * &self.dif
  }
}

impl<'a> DifElmat<'a, f64> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }
}

impl<'a, T> ElMatProviderBase for DifElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> ElMatProvider for DifElmat<'a, T> {
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    // This is only valid for the unweighted case
    debug_assert!(self.mass.weight.is_none());
    self._eval(geometry, None)
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CoordAwareElMatProvider
  for DifElmat<'a, T>
{
  fn eval_with_coords(&self, geometry: &SimplexLengths, topology: &Simplex) -> Matrix {
    self._eval(geometry, Some(topology))
  }
}

/// Element Matrix Provider for the weak mixed codifferential $(u, dif tau)$.
///
/// $A = [inner(lambda_J, dif lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_(k-1), J in Delta_k (K))$
pub struct CodifElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  mass: HodgeMassElmat<'a, T>,
  codif: Matrix,
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CodifElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  pub fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let mass = HodgeMassElmat::_new(dim, grade, coords, qr, weight);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade - 1);
    let dif = Matrix::from(&dif);
    let codif = dif.transpose();
    Self { mass, codif }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    let mass = self.mass._eval(geometry, topology);
    &self.codif * mass
  }
}

impl<'a> CodifElmat<'a, f64> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }
}

impl<'a, T> ElMatProviderBase for CodifElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CoordAwareElMatProvider
  for CodifElmat<'a, T>
{
  fn eval_with_coords(&self, geometry: &SimplexLengths, topology: &Simplex) -> Matrix {
    self._eval(geometry, Some(topology))
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> ElMatProvider for CodifElmat<'a, T> {
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    // This is only valid for the unweighted case
    debug_assert!(self.mass.weight.is_none());
    self._eval(geometry, None)
  }
}

/// Element Matrix Provider for the $(dif u, dif v)$ bilinear form.
///
/// $A = [inner(dif lambda_J, dif lambda_I)_(L^2 Lambda^(k+1) (K))]_(I,J in Delta_k (K))$
pub struct CodifDifElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  mass: HodgeMassElmat<'a, T>,
  dif: Matrix,
  codif: Matrix,
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CodifDifElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  pub fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let mass = HodgeMassElmat::_new(dim, grade + 1, coords, qr, weight);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade);
    let dif = Matrix::from(&dif);
    let codif = dif.transpose();

    Self { mass, dif, codif }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    let mass = self.mass._eval(geometry, topology);
    &self.codif * mass * &self.dif
  }
}

impl<'a> CodifDifElmat<'a, f64> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> ElMatProviderBase
  for CodifDifElmat<'a, T>
{
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CoordAwareElMatProvider
  for CodifDifElmat<'a, T>
{
  fn eval_with_coords(&self, geometry: &SimplexLengths, topology: &Simplex) -> Matrix {
    self._eval(geometry, Some(topology))
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> ElMatProvider for CodifDifElmat<'a, T> {
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    // This is only valid for the unweighted case
    debug_assert!(self.mass.weight.is_none());
    self._eval(geometry, None)
  }
}

pub type ElVec = Vector;
pub trait ElVecProvider: Sync {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec;
}

pub struct SourceElVec<'a, F, T>
where
  F: ExteriorField,
  T: ApplyWeight + AddAssign + Mul<f64, Output = T>,
{
  source: &'a F,
  mesh_coords: &'a MeshCoords,
  qr: SimplexQuadRule,
  weight: Option<&'a InnerProductWeightClosure<T>>,
}
impl<'a, F, T> SourceElVec<'a, F, T>
where
  F: ExteriorField,
  T: ApplyWeight + AddAssign + Mul<f64, Output = T>,
{
  pub fn new_weighted(
    source: &'a F,
    mesh_coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    Self::_new(source, mesh_coords, qr, Some(weight))
  }

  pub fn _new(
    source: &'a F,
    mesh_coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(source.dim_intrinsic()));
    Self {
      source,
      mesh_coords,
      qr,
      weight,
    }
  }
}

impl<'a, F> SourceElVec<'a, F, f64>
where
  F: ExteriorField,
{
  pub fn new(source: &'a F, mesh_coords: &'a MeshCoords, qr: Option<SimplexQuadRule>) -> Self {
    Self::_new(source, mesh_coords, qr, None)
  }
}
impl<F, T> ElVecProvider for SourceElVec<'_, F, T>
where
  F: Sync + ExteriorField,
  T: ApplyWeight + AddAssign + Mul<f64, Output = T>,
{
  fn grade(&self) -> ExteriorGrade {
    self.source.grade()
  }
  fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec {
    let cell_coords = SimplexCoords::from_simplex_and_coords(topology, self.mesh_coords);

    let dim = self.source.dim_intrinsic();
    let grade = self.grade();
    let dof_simps: Vec<_> = standard_subsimps(dim, grade).collect();
    let whitneys: Vec<_> = dof_simps
      .iter()
      .cloned()
      .map(|dof_simp| WhitneyLsf::standard(dim, dof_simp))
      .collect();

    let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), grade);

    let mut elvec = ElVec::zeros(whitneys.len());
    for (iwhitney, whitney) in whitneys.iter().enumerate() {
      let inner_pointwise = |local: CoordRef| {
        let global = cell_coords.local2global(local);
        let ref_source = self
          .source
          .at_point(&global)
          .precompose_form(&cell_coords.linear_transform());

        let source_coeffs = ref_source.coeffs();

        let weighted_owned = self
          .weight
          .map(|weight| weight.apply(global.as_view(), &source_coeffs));

        let weighted_source = weighted_owned.as_ref().unwrap_or(&source_coeffs);

        inner.inner(weighted_source, whitney.at_point(local).coeffs())
      };
      let value = self.qr.integrate_local(&inner_pointwise, geometry.vol());
      elvec[iwhitney] = value;
    }
    elvec
  }
}

#[cfg(test)]
mod test {
  use crate::operators::{
    CodifDifElmat, CodifElmat, CoordAwareElMatProvider, DifElmat, ElMatProvider, HodgeMassElmat,
    InnerProductWeightClosure, LaplaceBeltramiElmat, Matrix, ScalarMassElmat,
  };

  use approx::assert_relative_eq;
  use ddf::whitney::lsf::WhitneyLsf;
  use exterior::term::multi_gramian;
  use manifold::geometry::coord::mesh::MeshCoords;
  use manifold::topology::complex::Complex;
  use manifold::{geometry::metric::simplex::SimplexLengths, topology::simplex::standard_subsimps};

  #[test]
  fn codifdif0_is_laplace_beltrami() {
    let grade = 0;
    for dim in 1..=3 {
      let geo = SimplexLengths::standard(dim);
      let hodge_laplace = CodifDifElmat::new(dim, grade).eval(&geo);
      let laplace_beltrami = LaplaceBeltramiElmat::new(dim).eval(&geo);
      assert_relative_eq!(&hodge_laplace, &laplace_beltrami);
    }
  }

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    let grade = 0;
    for dim in 0..=3 {
      let geo = SimplexLengths::standard(dim);
      let hodge_mass = HodgeMassElmat::new(dim, grade).eval(&geo);
      let scalar_mass = ScalarMassElmat::new().eval(&geo);
      assert_relative_eq!(&hodge_mass, &scalar_mass);
    }
  }

  #[test]
  fn hodge_mass_dim2_grade1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = HodgeMassElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      1./3.,1./6.,0.   ;
      1./6.,1./3.,0.   ;
      0.   ,0.   ,1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn dif_n2_k1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = DifElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      -1./2., 1./3.,1./6.;
      -1./2., 1./6.,1./3.;
       0.   ,-1./6.,1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn codif_n2_k1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = CodifElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      -1./2., -1./2., 0.   ;
       1./3.,  1./6.,-1./6.;
       1./6.,  1./3., 1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn dif_dif_is_norm_of_difwhitneys() {
    for dim in 1..=3 {
      let geo = SimplexLengths::standard(dim);
      for grade in 0..dim {
        let difdif = CodifDifElmat::new(dim, grade).eval(&geo);

        let difwhitneys: Vec<_> = standard_subsimps(dim, grade)
          .map(|simp| WhitneyLsf::standard(dim, simp).dif())
          .collect();
        let mut inner = Matrix::zeros(difwhitneys.len(), difwhitneys.len());
        for (i, awhitney) in difwhitneys.iter().enumerate() {
          for (j, bwhitney) in difwhitneys.iter().enumerate() {
            inner[(i, j)] = multi_gramian(&geo.to_metric_tensor().inverse(), grade + 1)
              .inner(awhitney.coeffs(), bwhitney.coeffs());
          }
        }
        inner *= geo.vol();
        assert_relative_eq!(&difdif, &inner);
      }
    }
  }

  #[test]
  fn weighted_hodge_mass_scales_with_constant_scalar_weight() {
    const W: f64 = 2.5;
    const RTOL: f64 = 1e-12;

    for dim in 1..=3 {
      let geo = SimplexLengths::standard(dim);
      let topo = Complex::standard(dim);
      let cell = topo.cells().handle_iter().next().unwrap();

      for grade in 0..=dim {
        let unweighted = HodgeMassElmat::new(dim, grade).eval(&geo);

        let coords = MeshCoords::standard(dim);
        let weight = InnerProductWeightClosure::new(|_| W);

        let weighted = HodgeMassElmat::new_weighted(dim, grade, &coords, None, &weight)
          .eval_with_coords(&geo, &cell);

        let expected = W * &unweighted;
        assert_relative_eq!(&weighted, &expected, max_relative = RTOL);
      }
    }
  }

  #[test]
  fn weighted_hodge_mass_uses_cell_average_for_affine_weight_with_barycentric_qr() {
    // With barycentric quadrature and an affine weight w(x),
    // the implementation uses the (approx) cell-average, which matches w(barycenter).
    const RTOL: f64 = 1e-12;

    let dim = 2;
    let grade = 0;

    let geo = SimplexLengths::standard(dim);
    let topo = Complex::standard(dim);
    let cell = topo.cells().handle_iter().next().unwrap();

    // affine weight: w(x) = 1 + x0
    // on the standard simplex in R^dim, avg(x0) = 1/(dim+1), hence avg(w) = 1 + 1/(dim+1)
    let expected_w_avg = 1.0 + 1.0 / (dim as f64 + 1.0);

    let unweighted = HodgeMassElmat::new(dim, grade).eval(&geo);

    let coords = MeshCoords::standard(dim);
    let weight = InnerProductWeightClosure::new(|x| 1.0 + x[0]);

    let weighted = HodgeMassElmat::new_weighted(dim, grade, &coords, None, &weight)
      .eval_with_coords(&geo, &cell);

    let expected = expected_w_avg * &unweighted;
    assert_relative_eq!(&weighted, &expected, max_relative = RTOL);
  }

  #[test]
  fn weighted_hodge_mass_matrix_identity_matches_unweighted() {
    // sanity check that matrix-valued weights work and identity leaves the result unchanged
    const RTOL: f64 = 1e-12;

    let dim = 2;
    let grade = 1;

    let geo = SimplexLengths::standard(dim);
    let topo = Complex::standard(dim);
    let cell = topo.cells().handle_iter().next().unwrap();

    let unweighted = HodgeMassElmat::<f64>::new(dim, grade).eval(&geo);

    let coords = MeshCoords::standard(dim);

    // For 1-forms in 2D, coeff dimension is 2, so use 2x2 identity.
    let weight = InnerProductWeightClosure::new(|_| Matrix::identity(2, 2));

    let weighted = HodgeMassElmat::<Matrix>::new_weighted(dim, grade, &coords, None, &weight)
      .eval_with_coords(&geo, &cell);

    assert_relative_eq!(&weighted, &unweighted, max_relative = RTOL);
  }
}
