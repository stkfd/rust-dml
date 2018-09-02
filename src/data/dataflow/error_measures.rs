use data::serialization::AbomonableArray1;
use data::serialization::AsView;
use fnv::FnvHashMap;
use ndarray::prelude::*;
use num_traits::{cast::cast, Float};
use std::iter::Sum;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::{operators::Operator, Scope, Stream};
use timely::Data;

pub trait ErrorMeasure<F, E, D: Dimension> {
    fn error(prediction: &ArrayView<F, D>, original: &ArrayView<F, D>) -> E;
}

#[derive(Copy, Clone, Debug)]
pub struct Rmse;

impl<F: Float + Sum, D: Dimension> ErrorMeasure<F, F, D> for Rmse {
    fn error(prediction: &ArrayView<F, D>, original: &ArrayView<F, D>) -> F {
        assert_eq!(prediction.shape(), original.shape());
        let sum: F = original
            .iter()
            .zip(prediction.iter())
            .map(|(&expected, &actual)| {
                let diff = expected - actual;
                (diff * diff).sqrt()
            })
            .sum();
        sum / cast::<_, F>(prediction.len()).unwrap()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct IncorrectRatio;

impl<F: PartialEq, D: Dimension> ErrorMeasure<F, f64, D> for IncorrectRatio {
    fn error(prediction: &ArrayView<F, D>, original: &ArrayView<F, D>) -> f64 {
        assert_eq!(prediction.shape(), original.shape());
        let sum: u64 = original
            .iter()
            .zip(prediction.iter())
            .map(|(expected, actual)| {
                if expected != actual {
                    1_u64
                } else {
                    0_u64
                }
            })
            .sum();
        sum as f64 / prediction.len() as f64
    }
}

pub trait MeasurePredictionError<S: Scope, T, E> {
    fn prediction_error<M: ErrorMeasure<T, E, Ix1>>(
        &self,
        original: &Stream<S, AbomonableArray1<T>>,
        error_measure: M,
    ) -> Stream<S, E>;
}

impl<S: Scope, T: Data, E: Data> MeasurePredictionError<S, T, E> for Stream<S, AbomonableArray1<T>> {
    fn prediction_error<M: ErrorMeasure<T, E, Ix1>>(
        &self,
        original: &Stream<S, AbomonableArray1<T>>,
        _error_measure: M,
    ) -> Stream<S, E> {
        self.binary_frontier(original, Pipeline, Pipeline, "PredictionError", |_a, _b| {
            let mut stash = FnvHashMap::default();
            move |prediction_input, original_input, output| {
                prediction_input.for_each(|time, data| {
                    let entry = stash
                        .entry(time.retain())
                        .or_insert_with(|| (vec![], vec![]));
                    entry.0.extend(data.drain(..));
                });

                original_input.for_each(|time, data| {
                    let entry = stash
                        .entry(time.retain())
                        .or_insert_with(|| (vec![], vec![]));
                    entry.1.extend(data.drain(..));
                });

                let frontiers = &[prediction_input.frontier(), original_input.frontier()];
                for (time, (predictions, originals)) in &mut stash {
                    if frontiers.iter().all(|f| !f.less_equal(time)) {
                        let mut session = output.session(time);
                        for (prediction, original) in predictions.drain(..).zip(originals.drain(..))
                        {
                            session.give(M::error(&prediction.view(), &original.view()));
                        }
                    }
                }

                stash.retain(|_, (predictions, originals)| {
                    !predictions.is_empty() || !originals.is_empty()
                });
            }
        })
    }
}
