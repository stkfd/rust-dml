use super::*;
use std::ops::Add;
use timely::dataflow::Scope;
use timely::progress::PathSummary;
use timely::progress::Timestamp;

/// Configurable Source of randomly generated data
///
/// # Example
///
#[derive(Clone)]
pub struct RandRegressionTrainingSource<T1, T2, L, Dt, F> {
    samples_per_chunk: usize,
    chunks_per_round: usize,
    x_base_dist: Array1<Dt>,
    x_mapper: F,
    phantom: PhantomData<(T1, T2, L)>,
}

impl<T1, T2, L, Dt, F> RandRegressionTrainingSource<T1, T2, L, Dt, F>
where
    Dt: Clone,
    F: Fn(&ArrayView1<T1>, &mut ArrayViewMut1<T2>) -> L,
{
    pub fn new(x_mapper: F) -> Self {
        RandRegressionTrainingSource {
            samples_per_chunk: 100,
            chunks_per_round: 1,
            x_base_dist: Array1::from_shape_vec(0, vec![]).unwrap(),
            x_mapper,
            phantom: PhantomData,
        }
    }
}

impl<'a, T1, T2, L, Dt, F> RandRegressionTrainingSource<T1, T2, L, Dt, F>
where
    T1: Copy + Data + 'static,
    T2: Copy + Data + 'static,
    L: Add<Output = L> + Copy + Data + 'static,
    Dt: ToDistribution<T1> + 'static,
    F: Fn(&ArrayView1<T1>, &mut ArrayViewMut1<T2>) -> L + Clone + 'static,
{
    /// Configure the random distributions for the training input data
    pub fn x_distributions(mut self, base_dist: Array1<Dt>) -> Self {
        self.x_base_dist = base_dist;
        self
    }

    /// Configure the number of samples generated
    pub fn samples(self, samples_per_chunk: usize, chunks_per_round: usize) -> Self {
        RandRegressionTrainingSource {
            samples_per_chunk,
            chunks_per_round,
            ..self
        }
    }

    /// Create a stream that produces random data
    /// using the configuration in this instance
    pub fn to_stream<S: Scope>(
        &self,
        advance_by: <S::Timestamp as Timestamp>::Summary,
        end_when: S::Timestamp,
        scope: &S,
    ) -> Stream<S, TrainingData<T2, L>> {
        let base_dist = self.x_base_dist.map(|to_dist| to_dist.to_distribution());
        let chunks_per_round = self.chunks_per_round;
        let samples_per_chunk = self.samples_per_chunk;

        source(scope, "RandomSource", move |capability| {
            let mut cap = Some(capability);
            let x_mapper = self.x_mapper.clone();

            move |output| {
                let mut done = false;
                if let Some(cap) = cap.as_mut() {
                    {
                        let mut session = output.session(&cap);

                        for _ in 0..chunks_per_round {
                            // 2D Array to hold the final input values
                            let mut x = unsafe {
                                Array2::uninitialized((samples_per_chunk, base_dist.len()))
                            };
                            // Array to hold the output values
                            let mut y = unsafe { Array1::uninitialized(samples_per_chunk) };

                            // temporary array that holds each row's initial randomly generated values
                            let mut x_row_temp = unsafe { Array1::uninitialized(base_dist.len()) };

                            for (mut x_row, y) in x.outer_iter_mut().zip(y.iter_mut()) {
                                let mut rng = thread_rng();

                                for (x, dist) in x_row_temp.iter_mut().zip(base_dist.iter()) {
                                    *x = rng.sample(dist);
                                }
                                // give the supplied closure the random values and let it
                                // copy the final values into the 2d array
                                *y = x_mapper(&x_row_temp.view(), &mut x_row);
                            }

                            session.give(TrainingData {
                                x: x.into(),
                                y: y.into(),
                            });
                        }
                    }

                    if let Some(time) = advance_by.results_in(cap.time()) {
                        *cap = cap.delayed(&time);
                        done = time >= end_when;
                    } else {
                        done = true;
                    }
                }
                if done {
                    cap = None
                }
            }
        })
    }
}
