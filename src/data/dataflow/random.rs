use data::serialization::*;
use models::TrainingData;
use ndarray::prelude::*;
use rand::prelude::*;
use std::marker::PhantomData;
use timely::dataflow::operators::generic::source;
use timely::dataflow::scopes::Child;
use timely::dataflow::Scope;
use timely::dataflow::Stream;
use timely::Data;

/// Configurable Source of randomly generated data
pub struct RandomTrainingSource<T, L, D1, D2> {
    chunk_size: usize,
    rounds: usize,
    chunks_per_round: usize,
    x_dist: Array2<D1>,
    y_dist: Array1<D2>,
    phantom: PhantomData<(T, L)>,
}

impl<T, L, D1, D2> RandomTrainingSource<T, L, D1, D2>
where
    T: Copy + Data + 'static,
    L: Copy + Data + 'static,
    D1: Distribution<T> + 'static,
    D2: Distribution<L> + 'static,
{
    pub fn to_stream<'a, S: Scope>(
        self,
        scope: &Child<'a, S, u64>,
    ) -> Stream<Child<'a, S, u64>, TrainingData<T, L>> {
        source(scope, "RandomSource", move |capability| {
            let mut cap = Some(capability);
            move |output| {
                let mut done = false;
                if let Some(cap) = cap.as_mut() {
                    let mut session = output.session(&cap);
                    let rng = thread_rng();
                    for i in 0..self.chunks_per_round {
                        let mut x = unsafe { Array2::uninitialized(self.x_dist.dim()) };
                        let mut y = unsafe { Array1::uninitialized(self.y_dist.dim()) };
                        session.give(TrainingData { x: x.into(), y: y.into() });
                    }

                    let mut time = cap.time().clone();
                    time.inner += 1;
                    *cap = cap.delayed(&time);
                    //done = time.inner >= self.rounds as u64;
                }
                if done {
                    cap = None
                }
            }
        })
    }
}
