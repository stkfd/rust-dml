use num_traits::Float;
use std::cmp::Ordering;
use ndarray::prelude::*;
use ndarray_linalg::{Norm, Scalar};

/// Checks whether the K-Means Algorithm is converging.
pub trait ConvergenceCheck<T> {
    /// Check if the algorithm should abort, given the centroids from this and the previous iteration
    /// and the number of iterations
    fn converges<'a, 'b>(&self, old: &ArrayView2<'a, T>, new: &ArrayView2<'b, T>, iteration: usize) -> bool
        where T: 'a + 'b;
}

/// Convergence Criteria for the K-Means Algorithm.
#[derive(Default, Clone, Abomonation)]
pub struct ConvergenceCriteria<T> {
    pub max_iterations: Option<usize>,
    pub min_centroid_change: Option<T>
}

impl <T> ConvergenceCriteria<T> {
    /// Adds an iteration limit.
    pub fn limit_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = Some(iterations);
        self
    }

    /// Abort if the difference between the current and previous set of centroids
    /// is smaller than the given change.
    pub fn centroid_change(mut self, min_change: T) -> Self {
        self.min_centroid_change = Some(min_change);
        self
    }
}

impl <T> ConvergenceCheck<T> for ConvergenceCriteria<T>
    where T: Scalar + Float
{
    fn converges<'a, 'b>(&self, old: &ArrayView2<'a, T>, new: &ArrayView2<'b, T>, iteration: usize) -> bool where T: 'a + 'b {
        if let Some(max_iterations) = self.max_iterations {
            if max_iterations <= iteration { return true; }
        }

        if let Some(max_change) = self.min_centroid_change {
            // sum of euclidean distances between new and old centroids
            let distance = (new - old)
                .outer_iter()
                .map(|row| row.norm_l2())
                .collect::<Array<_,_>>()
                .scalar_sum();

            let cmp = distance.partial_cmp(&max_change.real()).unwrap_or(Ordering::Greater);
            return cmp == Ordering::Less || cmp == Ordering::Equal
        }

        false
    }
}