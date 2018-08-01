use data::serialization::AbomonableArray1;
use data::serialization::AsView;
use fnv::FnvHashMap;
use ndarray::prelude::*;
use num_traits::{cast::cast, Float};
use std::iter::Sum;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::{operators::Operator, Scope, Stream};
use timely::Data;

pub trait ErrorMeasure<F: Float, D: Dimension> {
    fn error(prediction: &ArrayView<F, D>, original: &ArrayView<F, D>) -> F;
}

#[derive(Copy, Clone, Debug)]
pub struct Rmse;

impl<F: Float + Sum, D: Dimension> ErrorMeasure<F, D> for Rmse {
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

pub trait MeasurePredictionError<S: Scope, T: Float> {
    fn prediction_error<M: ErrorMeasure<T, Ix1>>(
        &self,
        original: &Stream<S, AbomonableArray1<T>>,
        error_measure: M,
    ) -> Stream<S, T>;
}

impl<S: Scope, T: Data + Float> MeasurePredictionError<S, T> for Stream<S, AbomonableArray1<T>> {
    fn prediction_error<M: ErrorMeasure<T, Ix1>>(
        &self,
        original: &Stream<S, AbomonableArray1<T>>,
        _error_measure: M,
    ) -> Stream<S, T> {
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
